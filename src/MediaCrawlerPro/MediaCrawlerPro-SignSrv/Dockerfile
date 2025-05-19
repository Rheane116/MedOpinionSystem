FROM python:3.9.19-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake \
    pkg-config \
    libffi-dev \
    nodejs \
    npm \
    && pip3 install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*  # 清理APT缓存目录

COPY . .

CMD ["python", "app.py"]