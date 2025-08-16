# 🛒 Amazon 产品评价分析（Gemini） - Streamlit

本工具用于分析 Amazon 网店中某产品的用户评价。你可上传 CSV/Excel 文件（每行一条评价），应用将逐条调用 Google Gemini 提炼“优点/缺点”，并在你点击“生成汇总（流式输出）”后对整体结果进行归纳总结。

## 功能特性
- 上传 CSV/Excel（每行一条评价）
- 逐条调用 Gemini 提炼“优点/缺点”，实时展示
- 一键生成整体汇总（流式输出）
- 侧边栏可设置：
  - Gemini API Key
  - 模型（gemini-1.5-flash / gemini-1.5-pro）
  - 每条最多提炼条数（优/缺点各）

## 环境要求
- Python 3.9+（推荐 3.10/3.11）
- 可访问 Google Gemini API 的网络环境
- 一个有效的 Google Gemini API Key

## 安装与运行
1) 安装依赖
```bash
pip install -r requirements.txt
```

2) 配置 Gemini API Key（两选一）
- 环境变量（推荐）
  - macOS/Linux:
    ```bash
    export GOOGLE_API_KEY="你的API密钥"
    ```
  - Windows PowerShell:
    ```powershell
    $env:GOOGLE_API_KEY="你的API密钥"
    ```
- 或在应用左侧输入框直接粘贴

3) 启动应用
   
**方式一：直接运行（前台）**
```bash
streamlit run app.py
```
浏览器将自动打开（默认 http://localhost:8501）。如端口被占用，可指定端口：
```bash
streamlit run app.py --server.port 8502
```

**方式二：使用PM2后台运行（推荐生产环境）**

本项目已配置PM2支持，可在后台持续运行：

```bash
# 确保脚本有执行权限
chmod +x start_app.sh

# 使用PM2启动应用
pm2 start ecosystem.config.js

# 查看应用状态
pm2 status

# 查看应用日志
pm2 logs amazon-review-analyzer

# 停止应用
pm2 stop amazon-review-analyzer

# 重启应用
pm2 restart amazon-review-analyzer
```

注意：PM2方式运行时会使用`.james`虚拟环境，确保已正确创建该环境。

## 文件准备与格式
- 支持：CSV、XLSX、XLS
- 每行应是一条用户评价文本（可有多列，应用中选择哪一列是评价）
- CSV 建议 UTF-8 编码（程序对常见编码错误有兜底）

示例 CSV（UTF-8）：
```csv
review
很好用，续航长，做工精致
发热有点厉害，系统偶尔卡顿
外观漂亮，屏幕素质高，但价格稍贵
```

## 使用流程
1) 上传文件（CSV/Excel）
2) 选择“评价文本所在列”
3) （可选）在侧边栏设置模型与“每类最多提炼条数”
4) 点击“开始分析”：逐条提炼优点/缺点并实时展示
5) 点击“生成汇总（流式输出）”：对所有结果进行归纳总结（分组去重、排序，给出建议）

## 侧边栏配置说明
- Google Gemini API Key：可用环境变量 GOOGLE_API_KEY 或手动输入
- 模型选择：
  - gemini-1.5-flash：速度更快、成本更低
  - gemini-1.5-pro：能力更强但速度较慢
  - gemini-2.5-flash：最新模型，性能更优
- 每条最多提炼条数：限制每条评价的“优点/缺点”数量，避免冗余

## 隐私与成本
- 应用在本地运行，但会把评价文本通过 API 发送给 Gemini 模型用于分析
- 请避免上传含有敏感个人信息的数据
- 使用 Gemini 可能产生费用，请查看你的账户配额与计费

## 常见问题（FAQ）
- ModuleNotFoundError / openpyxl 错误
  ```bash
  pip install -r requirements.txt
  ```
- 403/401：API Key 无效或权限不足
  - 确认 Key 正确，启用了对应模型，区域/网络可访问
- CSV 中文乱码 / 编码问题
  - 尝试另存为 UTF-8
  - 或使用 Excel 格式（.xlsx/.xls）
- 端口被占用
  ```bash
  streamlit run app.py --server.port 8502
  ```
- 分析很慢
  - 选择 gemini-1.5-flash
  - 降低“每类最多提炼条数”

## 目录结构
```
.
├── app.py              # 主应用（Streamlit）
├── requirements.txt    # 依赖
└── README.md           # 使用说明（本文件）
```

## 许可证
仅供内部或学习使用。若用于商业，请自行评估合规与成本风险。