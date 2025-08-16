#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import glob
import math
import time
from typing import List, Dict

import pandas as pd

# Gemini
try:
    import google.generativeai as genai
except Exception as e:
    genai = None


MODEL_NAME = "gemini-2.5-flash"
BATCH_SIZE = 40  # 控制与模型交互的批大小，避免一次输入过大


def log(msg: str):
    print(msg, flush=True)


def input_api_key() -> str:
    env_key = os.getenv("GOOGLE_API_KEY", "").strip()
    prompt = "请输入 Google Gemini API Key（回车使用环境变量）: " if env_key else "请输入 Google Gemini API Key: "
    try:
        key = input(prompt).strip()
    except KeyboardInterrupt:
        log("\n已取消。")
        raise SystemExit(1)
    return key or env_key


def list_csv_files() -> List[str]:
    files = sorted(glob.glob("*.csv"))
    return files


def select_file(files: List[str]) -> str:
    if not files:
        log("当前目录未找到 CSV 文件。请将文件放到本目录后重试。")
        raise SystemExit(1)
    log(f"检测到 {len(files)} 个 CSV 文件，请输入编号选择：")
    for i, f in enumerate(files, start=1):
        log(f"  {i}. {f}")
    while True:
        s = input("请输入编号: ").strip()
        if not s.isdigit():
            log("请输入数字编号。")
            continue
        idx = int(s)
        if 1 <= idx <= len(files):
            return files[idx - 1]
        log("编号超出范围，请重试。")


def read_csv_smart(path: str) -> pd.DataFrame:
    # 多编码兜底
    encodings = ["utf-8", "utf-8-sig", "gbk", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            log(f"[读取] 使用编码 {enc} 读取成功，形状={df.shape}")
            return df
        except Exception as e:
            last_err = e
    raise RuntimeError(f"读取 CSV 失败：{last_err}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 去除列名两端空白与 BOM
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    log(f"[列名] 清洗后列名：{list(df.columns)}")
    return df


def to_float(x):
    if pd.isna(x):
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return 0.0
    # 去掉千分位、百分号
    s = s.replace(",", "")
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except Exception:
            return 0.0
    try:
        return float(s)
    except Exception:
        # 有些数据会用 "-" 或 "—" 表示空
        return 0.0


def configure_gemini(api_key: str):
    if genai is None:
        log("未找到 google-generativeai 库，请先安装：pip install google-generativeai")
        raise SystemExit(1)
    if not api_key:
        log("缺少 API Key。")
        raise SystemExit(1)
    genai.configure(api_key=api_key)
    sdk_ver = getattr(genai, "__version__", "unknown")
    log(f"[Gemini] 已配置，SDK版本={sdk_ver}，模型={MODEL_NAME}")


def heuristic_classify(text: str) -> str:
    """
    简单启发式：
    - ASIN：10位字母数字（常见以B0开头，但不强制）
    - keyword：包含空格的英文短语，或仅字母/空格/连字符/加号等
    返回 'ASIN' / 'keyword' / 'unknown'
    """
    t = (text or "").strip()
    if not t:
        return "unknown"

    t_no_space = re.sub(r"\s+", "", t)
    if re.fullmatch(r"[A-Z0-9]{10}", t_no_space):
        return "ASIN"

    # 如全是 ASCII，可认为倾向关键词；若包含中文/非ASCII，则多半不是英文关键词
    if all(ord(ch) < 128 for ch in t):
        # 如果是多个英文单词或包含常见连接符，判为关键词
        if re.fullmatch(r"[A-Za-z0-9\-\+\&\/\"\'\s\.\|:]+", t):
            # 如果像是单个无空格的 token 且近似 ASIN 格式但长度不等于10，仍不确定
            if (" " in t) or ("-" in t) or ("+" in t) or ("&" in t) or ("/" in t):
                return "keyword"
            # 单词但没有空格，默认仍当 keyword（如 iphone、case）
            return "keyword"

    return "unknown"


def gemini_classify_strings(texts: List[str]) -> List[str]:
    """
    使用 Gemini 批量判断字符串类型：'keyword' 或 'ASIN'
    """
    if not texts:
        return []
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config={"response_mime_type": "application/json"},
    )
    payload = {"items": texts}
    prompt = (
        "请判断以下字符串是“英文关键词”还是“ASIN”（亚马逊商品编号，通常为10位大写字母数字）。"
        "只允许两种输出：\"keyword\" 或 \"ASIN\"。按输入顺序返回JSON数组。\n"
        "输入示例: {\"items\": [\"iphone case\", \"B0C123ABCD\"]}\n"
        "输出示例: [\"keyword\", \"ASIN\"]\n\n"
        f"待判断数据：\n{json.dumps(payload, ensure_ascii=False)}"
    )
    try:
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        # 去掉可能的代码围栏
        if text.startswith("```"):
            text = text.strip("`")
            parts = text.split("\n", 1)
            if len(parts) == 2:
                text = parts[1]
            text = text.split("```")[0].strip()
        result = json.loads(text)
        if isinstance(result, list):
            out = []
            for v in result:
                v = str(v).strip().lower()
                if v not in ("keyword", "asin"):
                    out.append("keyword")  # 兜底
                else:
                    out.append("ASIN" if v == "asin" else "keyword")
            return out
    except Exception as e:
        log(f"[Gemini] classify 调用失败，使用启发式兜底：{e}")

    # 失败兜底：全部当 keyword
    return ["keyword"] * len(texts)


def classify_first_column_with_gemini(values: List[str]) -> List[str]:
    """
    综合：先启发式，unknown 的再走 Gemini；最终输出 'keyword'/'ASIN'
    """
    log(f"[分类] 共 {len(values)} 行，先进行启发式判断...")
    prelim = [heuristic_classify(v) for v in values]
    unknown_idx = [i for i, t in enumerate(prelim) if t == "unknown"]
    log(f"[分类] 启发式结果：ASIN={sum(1 for t in prelim if t=='ASIN')}, keyword={sum(1 for t in prelim if t=='keyword')}, unknown={len(unknown_idx)}")

    final_types = ["ASIN" if t == "ASIN" else "keyword" if t == "keyword" else None for t in prelim]

    if unknown_idx:
        to_check = [values[i] for i in unknown_idx]
        total = len(to_check)
        total_batches = math.ceil(total / BATCH_SIZE)
        log(f"[Gemini分类] 需要调用 Gemini 进一步判断 {total} 条，批大小={BATCH_SIZE}，批次数={total_batches}")
        classified = []
        for bi in range(0, total, BATCH_SIZE):
            chunk = to_check[bi : bi + BATCH_SIZE]
            batch_no = bi // BATCH_SIZE + 1
            log(f"[Gemini分类] 开始批 {batch_no}/{total_batches}，条目 {bi+1}-{min(bi+len(chunk), total)}")
            res = gemini_classify_strings(chunk)
            log(f"[Gemini分类] 完成批 {batch_no}/{total_batches}，返回 {len(res)} 条")
            classified.extend(res)
        for pos, v in zip(unknown_idx, classified):
            final_types[pos] = v

    # 兜底
    final_types = [v if v in ("keyword", "ASIN") else "keyword" for v in final_types]
    log(f"[分类] 最终统计：ASIN={sum(1 for t in final_types if t=='ASIN')}, keyword={sum(1 for t in final_types if t=='keyword')}")
    return final_types


def gemini_rate_keywords(rows: List[Dict]) -> List[Dict]:
    """
    rows: [{keyword, impressions, clicks, spend, orders, sales, acos, roas, ctr, cpc, conv_rate}]
    return: [{keyword, 产出评级, 优化建议}]
    """
    if not rows:
        return []

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config={"response_mime_type": "application/json"},
    )

    results: List[Dict] = []
    total = len(rows)
    total_batches = math.ceil(total / BATCH_SIZE)
    log(f"[Gemini评级] 需评级关键词 {total} 条，批大小={BATCH_SIZE}，批次数={total_batches}")

    for i in range(0, len(rows), BATCH_SIZE):
        chunk = rows[i : i + BATCH_SIZE]
        batch_no = i // BATCH_SIZE + 1
        log(f"[Gemini评级] 开始批 {batch_no}/{total_batches}，条目 {i+1}-{min(i+len(chunk), total)}")

        # 保证数字精简，减少token
        light = []
        for r in chunk:
            light.append({
                "keyword": r.get("keyword", ""),
                "imp": float(f'{r.get("impressions", 0.0):.4g}'),
                "clk": float(f'{r.get("clicks", 0.0):.4g}'),
                "spend": float(f'{r.get("spend", 0.0):.4g}'),
                "orders": float(f'{r.get("orders", 0.0):.4g}'),
                "sales": float(f'{r.get("sales", 0.0):.4g}'),
                "acos": float(f'{r.get("acos", 0.0):.4g}'),
                "roas": float(f'{r.get("roas", 0.0):.4g}'),
                "ctr": float(f'{r.get("ctr", 0.0):.4g}'),
                "cpc": float(f'{r.get("cpc", 0.0):.4g}'),
                "conv": float(f'{r.get("conv_rate", 0.0):.4g}'),
            })

        payload = {"rows": light}
        prompt = (
            "你是一名亚马逊广告数据分析师。请基于每行提供的指标，为每个关键词给出“产出评级”（高/中/低）"
            "以及简洁的“优化建议”（不超过30字）。指标含义：ROAS(越大越好)、ACOS(越小越好)、Orders、Sales、CTR、CPC、Conv等。\n"
            "请用以下参考：\n"
            "- 高：ROAS≥3 或 ACOS≤0.2，且有订单/销售；\n"
            "- 中：1≤ROAS<3 或 0.2<ACOS≤0.4，或有少量订单；\n"
            "- 低：ROAS<1 或 ACOS>0.4，或无订单高消耗；\n"
            "结合点击、转化率、CPC进行微调。\n"
            "只输出 JSON 数组，元素格式："
            "{\"keyword\":\"...\",\"产出评级\":\"高/中/低\",\"优化建议\":\"...\"}\n\n"
            f"输入：{json.dumps(payload, ensure_ascii=False)}"
        )
        try:
            resp = model.generate_content(prompt)
            text = (resp.text or "").strip()
            if text.startswith("```"):
                text = text.strip("`")
                parts = text.split("\n", 1)
                if len(parts) == 2:
                    text = parts[1]
                text = text.split("```")[0].strip()
            data = json.loads(text)
            if isinstance(data, list):
                results.extend(data)
                log(f"[Gemini评级] 完成批 {batch_no}/{total_batches}，累计返回 {len(results)} 条")
                continue
        except Exception as e:
            log(f"[Gemini评级] 批 {batch_no} 调用失败，使用规则兜底：{e}")

        # 失败兜底
        for r in chunk:
            roas = r.get("roas", 0.0)
            acos = r.get("acos", 1.0)
            orders = r.get("orders", 0.0)
            spend = r.get("spend", 0.0)
            if (roas >= 3 or acos <= 0.2) and orders > 0:
                lvl = "高"
                sug = "提升预算加大投放"
            elif (1 <= roas < 3) or (0.2 < acos <= 0.4) or orders > 0:
                lvl = "中"
                sug = "优化出价与词组匹配"
            else:
                lvl = "低"
                sug = "降低出价或暂停"
            results.append({"keyword": r.get("keyword", ""), "产出评级": lvl, "优化建议": sug})
        log(f"[Gemini评级] 批 {batch_no}/{total_batches} 使用兜底规则，累计返回 {len(results)} 条")
    return results


def gemini_summarize_keyword_suggestions(suggestions: List[str]) -> List[str]:
    if not suggestions:
        return []
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config={"response_mime_type": "application/json"},
    )
    # 去重截断
    uniq = []
    seen = set()
    for s in suggestions:
        s = (s or "").strip()
        if not s:
            continue
        if s not in seen:
            uniq.append(s)
            seen.add(s)
        if len(uniq) >= 100:
            break

    log(f"[建议汇总] 原始建议 {len(suggestions)} 条，去重后 {len(uniq)} 条")
    prompt = (
        "以下是针对关键词的多条优化建议，请去重、合并相似，输出3-6条可执行的行动建议。"
        "只输出 JSON 数组，元素为建议字符串。\n\n"
        f"建议列表：{json.dumps(uniq, ensure_ascii=False)}"
    )
    try:
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            parts = text.split("\n", 1)
            if len(parts) == 2:
                text = parts[1]
            text = text.split("```")[0].strip()
        arr = json.loads(text)
        if isinstance(arr, list):
            log(f"[建议汇总] Gemini 输出 {len(arr)} 条")
            return [str(x).strip() for x in arr if str(x).strip()]
    except Exception as e:
        log(f"[建议汇总] Gemini 调用失败，使用规则兜底：{e}")
    # 兜底
    return ["提高高产出词预算", "下调低产出词出价或否定", "优化词组匹配方式与否定词", "聚焦高转化词并扩展相近词"]


def main():
    start_ts = time.time()
    log("=== Amazon 广告 CSV 分析（Gemini 2.5 Flash）===")

    api_key = input_api_key()
    configure_gemini(api_key)

    files = list_csv_files()
    csv_path = select_file(files)
    log(f"[选择] 已选择文件：{csv_path}")

    df = read_csv_smart(csv_path)
    df = normalize_columns(df)

    # 数值列（按你的列名）
    # 先尝试标准列名，缺失则做模糊匹配
    def find_col(preferred: str, fallback_contains: str = None) -> str:
        if preferred in df.columns:
            return preferred
        if fallback_contains:
            for c in df.columns:
                if fallback_contains in c:
                    return c
        # 宽松：完全相等忽略空白
        for c in df.columns:
            if c.replace(" ", "") == preferred.replace(" ", ""):
                return c
        return preferred  # 返回预期名，后续如果不存在会 KeyError，更容易暴露问题

    # 关键列定位
    first_col = df.columns[0]  # 第一列即“匹配的商品”
    col_impr = find_col("展示量", "展示")
    col_clicks = find_col("点击次数", "点击")
    col_ctr = find_col("点击率 (CTR)", "点击率")
    col_spend = find_col("支出(USD)", "支出")
    col_cpc = find_col("单次点击成本 (CPC)(USD)", "CPC")
    col_orders = find_col("订单", "订单")
    col_sales = find_col("销售额(USD)", "销售额")
    col_acos = find_col("ACOS", "ACOS")
    col_roas = find_col("ROAS", "ROAS")
    col_conv = find_col("转化率", "转化")

    log("[列映射] 关键列对应：")
    log(f"  第一列: {first_col}")
    log(f"  展示量: {col_impr} | 点击次数: {col_clicks} | 点击率: {col_ctr}")
    log(f"  支出USD: {col_spend} | CPC: {col_cpc} | 订单: {col_orders} | 销售额USD: {col_sales}")
    log(f"  ACOS: {col_acos} | ROAS: {col_roas} | 转化率: {col_conv}")

    # 数值转换
    for c in [col_impr, col_clicks, col_ctr, col_spend, col_cpc, col_orders, col_sales, col_acos, col_roas, col_conv]:
        if c in df.columns:
            df[c] = df[c].map(to_float)
    log("[预处理] 数值列已转换完成。")

    # 1) 第一列分类 keyword / ASIN
    first_values = [str(x) if not pd.isna(x) else "" for x in df[first_col].tolist()]
    log(f"[步骤1] 开始判断第一列（共 {len(first_values)} 行）为 keyword 或 ASIN...")
    types = classify_first_column_with_gemini(first_values)
    df["标记"] = ["keyword" if t == "keyword" else "ASIN" for t in types]
    mark_counts = df["标记"].value_counts().to_dict()
    log(f"[步骤1] 完成，统计：{mark_counts}")

    # 2) 找出 支出(USD)>2 且 销售额(USD)=0 的 ASIN
    log("[步骤2] 筛选支出>2 且销售额=0 的 ASIN ...")
    asin_bad = []
    if col_spend in df.columns and col_sales in df.columns:
        mask = (df["标记"] == "ASIN") & (df[col_spend] > 2) & (df[col_sales] == 0)
        asin_bad = [str(v) for v in df.loc[mask, first_col].tolist() if str(v).strip()]
    asin_bad_str = ",".join(asin_bad)
    log(f"[结果] 命中 ASIN 数={len(asin_bad)}")
    log("[结果] 支出(USD)>2 且 销售额(USD)=0 的 ASIN：")
    log(asin_bad_str if asin_bad_str else "(无)")

    # 3) 对 keyword 行做产出评级与建议
    log("[步骤3] 对关键词行进行产出评级并生成建议 ...")
    keyword_rows = df[df["标记"] == "keyword"].copy()
    rate_inputs = []
    for _, r in keyword_rows.iterrows():
        rate_inputs.append({
            "keyword": str(r.get(first_col, "")).strip(),
            "impressions": float(r.get(col_impr, 0.0)),
            "clicks": float(r.get(col_clicks, 0.0)),
            "spend": float(r.get(col_spend, 0.0)),
            "orders": float(r.get(col_orders, 0.0)),
            "sales": float(r.get(col_sales, 0.0)),
            "acos": float(r.get(col_acos, 0.0)),
            "roas": float(r.get(col_roas, 0.0)),
            "ctr": float(r.get(col_ctr, 0.0)),
            "cpc": float(r.get(col_cpc, 0.0)),
            "conv_rate": float(r.get(col_conv, 0.0)),
        })

    rated = gemini_rate_keywords(rate_inputs)
    log(f"[步骤3] 评级完成，返回条数={len(rated)}")

    # 合并回 DataFrame
    rate_map = {x.get("keyword", ""): (x.get("产出评级", ""), x.get("优化建议", "")) for x in rated}
    df["产出评级"] = ""
    df["优化建议"] = ""
    suggestions_all = []
    for idx, r in df.iterrows():
        key = str(r.get(first_col, "")).strip()
        if r.get("标记") == "keyword" and key in rate_map:
            lvl, sug = rate_map[key]
            df.at[idx, "产出评级"] = str(lvl or "")
            df.at[idx, "优化建议"] = str(sug or "")
            if sug:
                suggestions_all.append(str(sug))

    lvl_counts = df["产出评级"].value_counts().to_dict()
    log(f"[步骤3] 产出评级分布：{lvl_counts if lvl_counts else '(无)'}")

    # 汇总关键词优化建议（全局）
    merged_suggestions = gemini_summarize_keyword_suggestions(suggestions_all)
    log("\n[关键词优化建议 - 汇总]")
    for i, s in enumerate(merged_suggestions, start=1):
        log(f"{i}. {s}")

    # 输出文件
    out_path = f"analyzed_{os.path.basename(csv_path)}"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    elapsed = time.time() - start_ts
    log(f"\n[输出] 已生成结果文件：{out_path}")
    log("[输出] 新增列：['标记', '产出评级', '优化建议']")
    log(f"[完成] 总耗时：{elapsed:.2f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n已中断。")