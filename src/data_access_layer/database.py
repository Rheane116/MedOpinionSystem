from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# ORM数据库连接
db = SQLAlchemy()

def init_db(app):
  # 初始化ORM数据库
  db.init_app(app)
    # 创建绑定多个数据库的引擎
  app.config['SQLALCHEMY_BINDS'] = {
      'sys': app.config['DB_SYS_URI'],
      'crawl': app.config['DB_CRAWL_URI']
  }
  with app.app_context():
    with app.app_context():
      # 创建默认数据库表
      db.create_all()

def get_session(bind_key=None):
    """获取数据库会话"""
    if bind_key:
        engine = create_engine(db.get_engine(bind=bind_key))
    else:
        engine = db.engine
    return sessionmaker(bind=engine)() 

