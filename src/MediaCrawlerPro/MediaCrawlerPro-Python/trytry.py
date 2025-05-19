'''import redis

# 创建 Redis 连接
r = redis.Redis(
    host='172.17.0.8',  # Redis 容器的 IP 地址
    port=6379,         # Redis 端口
    password='root',  # Redis 密码（如果设置了密码）
    db=0               # 数据库编号
)

# 测试连接
print(r.ping())  # 输出: True'''

from utils.filter import crawl_filter
from utils.query import *

task_args = {
        'task_id' : 0,
        'platforms' : '知乎',
        #'keywords' :  keywords,
        'agency' : '深圳市人民医院',
        'all_names' : '深圳人民医院',
        'max_note' : '200',
        'max_comment' : '100',
        'start_time' : '03/04/2020',
        'end_time' : '/03/24/2025',
        'status' : '5',
        'gpu' : '0'
    }
crawl_filter(task_args)
