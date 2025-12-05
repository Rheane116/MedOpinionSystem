# 声明：本代码仅供学习和研究目的使用。使用者应遵守以下原则：  
# 1. 不得用于任何商业用途。  
# 2. 使用时应遵守目标平台的使用条款和robots.txt规则。  
# 3. 不得进行大规模爬取或对平台造成运营干扰。  
# 4. 应合理控制请求频率，避免给目标平台带来不必要的负担。   
# 5. 不得用于任何非法或不当的用途。
#   
# 详细许可条款请参阅项目根目录下的LICENSE文件。  
# 使用本代码即表示您同意遵守上述原则和LICENSE中的所有条款。  


import os

# mysql config
RELATION_DB_PWD = os.getenv("RELATION_DB_PWD", "root")
RELATION_DB_USER = os.getenv("RELATION_DB_USER", "root")
RELATION_DB_HOST = os.getenv("RELATION_DB_HOST", "10.249.41.226")
RELATION_DB_PORT = os.getenv("RELATION_DB_PORT", "5508")
RELATION_DB_NAME = os.getenv("RELATION_DB_NAME", "data_fuwai")

RELATION_DB_URL = f"mysql://{RELATION_DB_USER}:{RELATION_DB_PWD}@{RELATION_DB_HOST}:{RELATION_DB_PORT}/{RELATION_DB_NAME}"

# redis config
REDIS_DB_HOST = os.getenv("REDIS_DB_HOST", "172.17.0.8")  # your redis host
REDIS_DB_PWD = os.getenv("REDIS_DB_PWD", "root")  # your redis password
REDIS_DB_PORT = os.getenv("REDIS_DB_PORT", 6379)  # your redis port
REDIS_DB_NUM = os.getenv("REDIS_DB_NUM", 0)  # your redis db num

# cache type
CACHE_TYPE_REDIS = "redis"
CACHE_TYPE_MEMORY = "memory"
USE_CACHE_TYPE = CACHE_TYPE_REDIS # 请使用 redis 作为缓存，因为这样不会浪费代理IP，本地换成如果程序重启，代理IP就会丢失
