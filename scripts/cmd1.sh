#!/bin/bash


#echo "Starting docker mysql..."
#docker start mysql_zyy



# 等待 Docker 容器完全启动（可根据需要调整等待时间）
#echo "Waiting for docker mysql to start..."
#sleep 2

# 启动 Docker 容器
#echo "Starting docker crawler_sign..."
    # -d: detached表示后台运行，不会占用当前终端
#docker run -d --name crawler_sign_zyy -p 8989:8989 --restart=always -e LOGGER_LEVEL=ERROR mediacrawler_signsrv
#docker start crawler_sign_zyy

# 在 Conda 环境中运行 Python 脚本s
echo "Running Medical-Public-Opinion-System..." 
#conda run --no-capture-output -n base python ./flaskPro/app.py 
#echo "请在log/system.log中查看系统状态日志以及网页url~"

python -m  src.web_layer.app1
#conda run --no-capture-output -n base python ./flaskPro/app.py > ./logs/system.log 2>&1 

# 停止 Docker 容器
#echo "Stopping Docker crawler_sign..."
#docker stop crawler_sign_zyy





