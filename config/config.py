import os
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'this is a secret key'
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:root@localhost:5508/med_system'
    DB_SYS_URI = 'mysql+pymysql://root:root@localhost:5508/med_system'
    DB_CRAWL_URI = 'mysql+pymysql://root:root@localhost:5508/media_crawler'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_BINDS = {
        'sys': DB_SYS_URI,
        'crawl': DB_CRAWL_URI
    }  