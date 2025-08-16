import os
import json
from typing import List, Tuple

import pandas as pd
import streamlit as st
import time
import random

# Google Gemini
import google.generativeai as genai

# 抑制 Tornado 在客户端断开时的 WebSocketClosedError 噪音，并降低 Tornado 日志级别
try:
    import asyncio
    import logging
    from tornado.websocket import WebSocketClosedError
    from tornado.iostream import StreamClosedError

    def _ignore_ws_closed(loop, context):
        err = context.get("exception")
        if isinstance(err, (WebSocketClosedError, StreamClosedError)):
            # 忽略客户端断开连接导致的噪音异常
            return
        loop.default_exception_handler(context)

    try:
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(_ignore_ws_closed)
    except Exception:
        # 某些运行环境下事件循环不可设置，安全忽略
        pass

    for _logger_name in ("tornado.general", "tornado.access", "tornado.application", "tornado.websocket"):
        logging.getLogger(_logger_name).setLevel(logging.ERROR)
except Exception:
    # 安全兜底，避免影响应用主流程
    pass


# -----------------------
# 页面与基础设置
# -----------------------
st.set_page_config(page_title="Amazon 评价分析 - Gemini", page_icon="🛒", layout="wide")

st.title("🛒 Amazon 产品评价分析（Gemini）")
st.caption("上传 CSV/Excel（每行一条评价），逐条提炼优缺点，并支持一键生成汇总（流式输出）")


# -----------------------
# 侧边栏设置
# -----------------------
with st.sidebar:
    st.header("设置")
    default_key = os.getenv("GOOGLE_API_KEY", "")
    api_key = st.text_input("Google Gemini API Key", value=default_key, type="password", help="优先使用环境变量 GOOGLE_API_KEY")
    model_name = st.selectbox("选择模型", ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-flash"], index=0, help="Flash 更快更省，Pro 更强但更慢，2.5 Flash 是最新模型")
    max_items = st.slider("每条评价最多提炼条数（优/缺点各）", 1, 8, 5, 1)
    rpm = st.slider("每分钟最大请求数（RPM）", 5, 120, 30, 5)
    st.session_state["_rpm"] = rpm
    st.markdown("---")
    st.info("提示：请确保网络可访问 Gemini 服务。如需安全保管密钥，建议使用环境变量 GOOGLE_API_KEY。")


# -----------------------
# 会话状态
# -----------------------
if "analysis_results" not in st.session_state:
    # List[dict]: {index, review, pros[List[str]], cons[List[str]]}
    st.session_state.analysis_results = []
if "analysis_params" not in st.session_state:
    st.session_state.analysis_params = {}
if "summary_done" not in st.session_state:
    st.session_state.summary_done = False


# -----------------------
# 工具函数
# -----------------------
def init_gemini(_api_key: str):
    if not _api_key:
        raise ValueError("缺少 API Key")
    genai.configure(api_key=_api_key)


def read_dataframe(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        # 简单编码兜底
        try:
            df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="latin1")
        return df
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    st.error("仅支持 CSV、XLSX 或 XLS 文件")
    return pd.DataFrame()


def guess_text_columns(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return []
    # 优先 object 类型，兜底全部列
    text_cols = list(df.select_dtypes(include=["object"]).columns)
    if not text_cols:
        text_cols = list(df.columns)
    return text_cols


def unique_preserve(seq: List[str], limit: int) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        x = (x or "").strip()
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
        if len(out) >= limit:
            break
    return out


def build_extract_prompt(review: str, max_items_each: int) -> str:
    return f"""
你是一个中立的产品评价分析助手。请从以下用户评价中提炼该产品的优点与缺点，要求：
- 只关注产品本身的特性与体验；除非直接影响产品体验，否则忽略物流/客服/包装/价格波动等与产品无关的信息。
- 每类最多提炼 {max_items_each} 条；短语形式、去重、用简体中文。
- 若无明确信息，可返回空列表。

只输出 JSON，格式如下：
{{
  "pros": ["优点1", "优点2", "..."],
  "cons": ["缺点1", "缺点2", "..."]
}}

用户评价：
\"\"\"{review}\"\"\"
""".strip()


def analyze_one_review(review: str, model_name: str, max_items_each: int, rpm: int) -> Tuple[List[str], List[str]]:
    # 带速率限制与指数退避重试，降低429概率
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config={"response_mime_type": "application/json"},
    )
    prompt = build_extract_prompt(review, max_items_each)
    retries = 4
    base_delay = 2.0
    for attempt in range(retries + 1):
        try:
            # 速率限制：控制每分钟请求数
            min_interval = 60.0 / max(1, rpm)
            last_ts = st.session_state.get("_last_call_ts", 0.0)
            elapsed = time.time() - last_ts
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            resp = model.generate_content(prompt)
            st.session_state["_last_call_ts"] = time.time()

            text = (resp.text or "").strip()
            # 去掉可能的代码围栏
            if text.startswith("```"):
                text = text.strip("`")
                # 处理例如 ```json ... ```
                parts = text.split("\n", 1)
                if len(parts) == 2:
                    text = parts[1]
                text = text.split("```")[0].strip()

            data = json.loads(text)
            pros = data.get("pros", [])
            cons = data.get("cons", [])
            # 规范化
            pros = [str(x).strip() for x in pros if str(x).strip()]
            cons = [str(x).strip() for x in cons if str(x).strip()]
            pros = unique_preserve(pros, max_items_each)
            cons = unique_preserve(cons, max_items_each)
            return pros, cons
        except Exception as e:
            msg = str(e).lower()
            # 针对配额/速率限制做指数退避
            if ("429" in msg) or ("quota" in msg) or ("rate" in msg) or ("exceed" in msg):
                delay = base_delay * (1.6 ** attempt) + random.uniform(0, 0.5)
                # 简单等待后重试
                time.sleep(delay)
                continue
            # 其他错误也做有限重试
            if attempt < retries:
                time.sleep(1.0 + 0.25 * attempt)
                continue
            # 最终失败
            return [], [f"解析失败或调用错误：{e}"]


def build_summary_prompt(all_pros: List[str], all_cons: List[str]) -> str:
    pros_block = "\n".join([f"- {p}" for p in all_pros[:300]])  # 控制长度
    cons_block = "\n".join([f"- {c}" for c in all_cons[:300]])
    return f"""
你将基于以下来自用户评价提炼的“优点/缺点”进行产品级总结，请：
- 去重与归类（合并表达相似的点），按重要性排序；
- 输出结构：
  1) 关键优点（分组，组名+1-2句说明）
  2) 关键不足（分组，组名+1-2句说明）
  3) 结论与建议（3-5条，面向产品优化与营销策略，可执行）
- 用简体中文，避免重复和空话。

优点（样本）：
{pros_block}

缺点（样本）：
{cons_block}
""".strip()


def stream_summary(all_pros: List[str], all_cons: List[str], model_name: str, rpm: int):
    # 流式生成，带速率限制与有限重试
    model = genai.GenerativeModel(model_name=model_name)
    prompt = build_summary_prompt(all_pros, all_cons)
    retries = 2
    base_delay = 2.0
    for attempt in range(retries + 1):
        try:
            # 速率限制：控制每分钟请求数
            min_interval = 60.0 / max(1, rpm)
            last_ts = st.session_state.get("_last_call_ts", 0.0)
            elapsed = time.time() - last_ts
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            response = model.generate_content(prompt, stream=True)
            st.session_state["_last_call_ts"] = time.time()
            for chunk in response:
                try:
                    if hasattr(chunk, "text") and chunk.text:
                        yield chunk.text
                except Exception:
                    continue
            return
        except Exception as e:
            msg = str(e).lower()
            if ("429" in msg) or ("quota" in msg) or ("rate" in msg) or ("exceed" in msg):
                delay = base_delay * (1.6 ** attempt) + random.uniform(0, 0.5)
                yield f"\n\n（等待配额恢复 {delay:.1f}s 后重试...）"
                time.sleep(delay)
                continue
            else:
                yield f"\n\n（生成失败：{e}）"
                return
    yield "\n\n（多次重试后仍受限，请稍后再试。）"


# -----------------------
# 上传与列选择
# -----------------------
uploaded = st.file_uploader("上传文件（CSV/Excel，每行一条评价）", type=["csv", "xlsx", "xls"])

df = read_dataframe(uploaded)
if not df.empty:
    st.subheader("数据预览")
    st.dataframe(df.head(20), use_container_width=True)

col_options = guess_text_columns(df) if not df.empty else []
selected_col = None
if col_options:
    selected_col = st.selectbox("选择评价文本所在列", col_options, index=0)

# -----------------------
# 分析按钮与过程
# -----------------------
col_btn1, col_btn2 = st.columns([1, 1])

start_clicked = col_btn1.button("开始分析", type="primary", disabled=(df.empty or selected_col is None))
summary_clicked = col_btn2.button("生成汇总（流式输出）", disabled=(len(st.session_state.analysis_results) == 0))

results_placeholder = st.empty()
progress_placeholder = st.empty()
status_placeholder = st.empty()
summary_placeholder = st.empty()

if start_clicked:
    st.session_state.summary_done = False

    if not api_key:
        st.error("请在左侧输入 Google Gemini API Key（或设置环境变量 GOOGLE_API_KEY）")
    else:
        try:
            init_gemini(api_key)
        except Exception as e:
            st.error(f"初始化 Gemini 失败：{e}")
        else:
            # 清空旧结果
            st.session_state.analysis_results = []

            # 清洗文本
            reviews_series = df[selected_col].fillna("").astype(str).map(lambda x: x.strip())
            reviews = [r for r in reviews_series.tolist() if r]

            total = len(reviews)
            if total == 0:
                st.warning("未找到有效的评价文本。")
            else:
                progress = progress_placeholder.progress(0)
                table_container = results_placeholder.container()

                for i, review in enumerate(reviews, start=1):
                    pros, cons = analyze_one_review(review, model_name, max_items, st.session_state.get("_rpm", 30))
                    st.session_state.analysis_results.append({
                        "index": i,
                        "review": review,
                        "pros": pros,
                        "cons": cons
                    })

                    # 展示当前累计结果（表格形式）
                    display_rows = []
                    for row in st.session_state.analysis_results:
                        display_rows.append({
                            "序号": row["index"],
                            "评价": row["review"],
                            "优点": "；".join(row["pros"]) if row["pros"] else "",
                            "缺点": "；".join(row["cons"]) if row["cons"] else "",
                        })
                    table_container.dataframe(pd.DataFrame(display_rows), use_container_width=True, height=450)

                    progress.progress(i / total)
                    status_placeholder.info(f"已分析 {i}/{total} 条")

                status_placeholder.success("分析完成。可点击“生成汇总（流式输出）”")

if summary_clicked:
    if not api_key:
        st.error("请在左侧输入 Google Gemini API Key（或设置环境变量 GOOGLE_API_KEY）")
    elif len(st.session_state.analysis_results) == 0:
        st.warning("请先完成逐条分析。")
    else:
        try:
            init_gemini(api_key)
        except Exception as e:
            st.error(f"初始化 Gemini 失败：{e}")
        else:
            # 汇总去重
            all_pros: List[str] = []
            all_cons: List[str] = []
            for row in st.session_state.analysis_results:
                all_pros.extend(row.get("pros", []))
                all_cons.extend(row.get("cons", []))

            # 去重并保序，避免提示过长
            all_pros = unique_preserve(all_pros, limit=1000)
            all_cons = unique_preserve(all_cons, limit=1000)

            st.subheader("🧾 汇总（流式输出）")
            with st.container(border=True):
                # 流式输出
                st.write_stream(stream_summary(all_pros, all_cons, model_name, st.session_state.get("_rpm", 30)))
            st.session_state.summary_done = True


# -----------------------
# 底部说明
# -----------------------
with st.expander("使用说明"):
    st.markdown(
        "- 上传 CSV/Excel，选择评价所在列；\n"
        "- 点击“开始分析”逐条提炼优缺点（会实时显示每条的分析结果）；\n"
        "- 完成后点击“生成汇总（流式输出）”，得到聚合后的产品优缺点、结论与建议；\n"
        "- 如需更快或更强分析，可在侧边栏切换模型。"
    )