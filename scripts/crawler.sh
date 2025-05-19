#!/bin/bash




# 等待 Docker 容器完全启动（可根据需要调整等待时间）
#echo "Waiting for dockers to start..."
#sleep 2

# 在 Conda 环境中运行 Python 脚本
echo "Running Crawler..."
echo "请在log/crawler.log中查看数据爬取进度~"
conda run --no-capture-output -n crawler python ./src/MediaCrawlerPro/MediaCrawlerPro-Python/main.py \
            --platform $1 --keywords $2  --max_note $3 --max_comment $4 \
            --if_level $5 --task_id $6 --username $7\
            > ./logs/crawler.log 2>&1


# 获取上一个命令的退出状态
EXIT_STATUS=$?
 

# 判断退出状态
if [ $EXIT_STATUS -eq 0 ]; then
    echo "程序正确退出，日志已保存到 logs/crawler.log"
else
    echo "程序异常退出，退出状态码：$EXIT_STATUS, 日志已保存到 logs/crawler.log"
    exit $EXIT_STATUS
fi



#echo "All tasks completed!"