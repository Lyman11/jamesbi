#!/bin/bash

# 激活Python虚拟环境
source .james/bin/activate

# 启动Streamlit应用
streamlit run app.py --server.port=8501 --server.address=0.0.0.0