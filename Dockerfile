
FROM python:3.9-slim

# # 安装系统依赖，含构建工具（解决 spaCy / blis 构建问题）
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

    # 设置工作目录
WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 下载英文小模型
RUN python -m spacy download en_core_web_sm

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000
EXPOSE 8501 
EXPOSE 11434

# 用 bash 运行 run.sh
CMD ["bash", "run.sh"]
