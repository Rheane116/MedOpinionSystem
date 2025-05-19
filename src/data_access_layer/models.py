from .database import db
import hashlib
import time

def get_password_hash(password: str):
    return hashlib.sha256(password.encode()).hexdigest()

class User(db.Model):
    __bind_key__ = 'sys'
    __table_name__ = 'user'

    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(64), nullable=False)
    create_time = db.Column(db.String(20), nullable=True)
    role = db.Column(db.Integer, nullable=True, default=0)
    phone = db.Column(db.String(15), nullable=True)
    email = db.Column(db.String(30), nullable=True)
    agency = db.Column(db.String(30), nullable=True)


    # 定义模型之间的关系（一对一、一对多或者多对多）
    #                    关联的模型类名； 在Task中自动创建的反向引用属性； 关联数据的加载方式，True表示延迟加载
    tasks = db.relationship('Task', backref='user', lazy=True)

    def set_password(self, password):
        self.password = get_password_hash(password)

    def check_password(self, pwd):
        return get_password_hash(pwd) == self.password 
  
class Task(db.Model):
    __bind_key__ = 'sys'
    __table_name__ = 'task'    

    task_id = db.Column(db.Integer, primary_key=True)
    agency_name = db.Column(db.String(30), nullable=False)
    other_names = db.Column(db.String(100), nullable=True)
    platform = db.Column(db.String(40), nullable=False)

    status = db.Column(db.Integer, nullable=False, default=0)
    user_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), nullable=False)


    start_time = db.Column(db.String(10), nullable=False)
    end_time = db.Column(db.String(10), nullable=False)
    create_time = db.Column(db.String(20), nullable=True)

    max_note = db.Column(db.Integer, nullable=True, default=300)
    max_comment = db.Column(db.Integer, nullable=True, default=100)
    if_level = db.Column(db.Integer, nullable=True, default=0)

    gpu = db.Column(db.String(10), nullable=True, default='0')
    batch_size = db.Column(db.Integer, nullable=True, default=8)
    result = db.Column(db.Text, nullable=True)
    data_num = db.Column(db.Integer, nullable=True, default=0)

    def to_dict(self):
        return{
            "task_id": self.task_id,
            "agency_name": self.agency_name,
            "other_names": self.other_names,
            "platform": self.platform,
            "status": self.status,
            "user_id": self.user_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "create_time": self.create_time,
            "max_note": self.max_note,
            "max_comment": self.max_comment,
            "if_level": self.if_level,
            "gpu": self.gpu,
            "batch_size": self.batch_size,
            "result": self.result,
            "user_id": self.user_id
        }

class TaskManual(db.Model):
    __bind_key__ = 'sys'
    __table_name__ = 'task_manual'    

    task_id = db.Column(db.Integer, primary_key=True)
    agency_name = db.Column(db.String(30), nullable=False)
    other_names = db.Column(db.String(100), nullable=True)
    status = db.Column(db.Integer, nullable=False, default=0)
    user_id = db.Column(db.Integer, nullable=False)
    create_time = db.Column(db.String(20), nullable=True)

    gpu = db.Column(db.String(10), nullable=True, default='0')
    batch_size = db.Column(db.Integer, nullable=True, default=8)

    sep = db.Column(db.String(20), nullable=True)
    input_format = db.Column(db.String(4), nullable=False)
    row = db.Column(db.Integer, nullable=True)
    col = db.Column(db.Integer, nullable=True)
    result = db.Column(db.Text, nullable=True)

    def to_dict(self):
        return{
            "task_id": self.task_id,
            "agency_name": self.agency_name,
            "other_names": self.other_names,
            "status": self.status,
            "user_id": self.user_id,
            "create_time": self.create_time,
            "gpu": self.gpu,
            "batch_size": self.batch_size,
            "sep": self.sep,
            "input_format": self.input_format,
            "row": self.row,
            "col": self.col,
            "result": self.result
        }

class TaskData(db.Model):
    __bind_key__ = 'sys'
    __table_name__ = 'task_data'

    data_id = db.Column(db.String(20), primary_key=True)
    platform = db.Column(db.String(10), nullable=False)
    task_id = db.Column(db.Integer, db.ForeignKey('task.task_id'), nullable=False)
    time = db.Column(db.String(20), nullable=True)
    text = db.Column(db.Text, nullable=True)
    output = db.Column(db.Text, nullable=True)

class TaskDataManual(db.Model):
    __bind_key__ = 'sys'
    __table_name__ = 'task_data_manual'

    data_id = db.Column(db.String(10), primary_key=True)
    task_id = db.Column(db.Integer, db.ForeignKey('task_manual.task_id'), nullable=False)
    text = db.Column(db.Text, nullable=True)
    output = db.Column(db.Text, nullable=True)

class GlobalConfig(db.Model):
    __bind_key__ = 'sys'
    __table_name__ = 'global_config'

    user_id = db.Column(db.Integer, primary_key=True)
    agency_name = db.Column(db.String(30), nullable=True)
    other_names = db.Column(db.String(100), nullable=True)
    max_note = db.Column(db.Integer, nullable=True)
    max_comment = db.Column(db.Integer, nullable=True)
    gpu = db.Column(db.String(10), nullable=True)
    batch_size = db.Column(db.Integer, nullable=True)
    if_level = db.Column(db.Integer, nullable=True)

    def to_dict(self):
        return {
            'agency_name': self.agency_name,
            'other_names': self.other_names,
            'max_note': self.max_note,
            'max_comment': self.max_comment,
            'if_level': self.if_level,
            'gpu': self.gpu,
            'batch_size': self.batch_size
        }

class CookieAccount(db.Model):
    __bind_key__ = 'crawl'
    __tablename__ = 'crawler_cookies_account'

    id = db.Column(db.Integer, primary_key=True)
    account_name = db.Column(db.String(64), nullable=False)
    platform_name = db.Column(db.String(10), nullable=False)
    cookies = db.Column(db.Text, nullable=True)
    create_time = db.Column(db.DateTime, nullable=True)
    update_time = db.Column(db.DateTime, nullable=True)
    invalid_timestamp = db.Column(db.Integer, nullable=False, default=0)
    status = db.Column(db.Integer, nullable=False, default=0)

class ZhihuComment(db.Model):
    __bind_key__ = 'crawl'
    __tablename__ = 'zhihu_comment'

    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    create_time = db.Column(db.String(32), nullable=False, name="publish_time")
    task_id = db.Column(db.Integer, nullable=True)

class XhsComment(db.Model):
    __bind_key__ = 'crawl'
    __tablename__ = 'xhs_note_comment'

    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    create_time = db.Column(db.Integer, nullable=False)
    task_id = db.Column(db.Integer, nullable=True)

class BilibiliComment(db.Model):
    __bind_key__ = 'crawl'
    __tablename__ = 'bilibili_video_comment'

    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    create_time = db.Column(db.Integer, nullable=False)
    task_id = db.Column(db.Integer, nullable=True)

class WeiboComment(db.Model):
    __bind_key__ = 'crawl'
    __tablename__ = 'weibo_note_comment'

    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    create_time = db.Column(db.Integer, nullable=False)
    task_id = db.Column(db.Integer, nullable=True)

class ZhihuContent(db.Model):
    __bind_key__ = 'crawl'
    __tablename__ = 'zhihu_content'

    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False, name="desc")
    create_time = db.Column(db.String(32), nullable=False, name="updated_time")
    task_id = db.Column(db.Integer, nullable=True)

class XhsContent(db.Model):
    __bind_key__ = 'crawl'
    __tablename__ = 'xhs_note'

    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False, name="desc")
    create_time = db.Column(db.Integer, nullable=False, name="last_update_time")
    task_id = db.Column(db.Integer, nullable=True)

class BilibiliContent(db.Model):
    __bind_key__ = 'crawl'
    __tablename__ = 'bilibili_video'

    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False, name="desc")
    create_time = db.Column(db.Integer, nullable=False)
    task_id = db.Column(db.Integer, nullable=True)

class WeiboContent(db.Model):
    __bind_key__ = 'crawl'
    __tablename__ = 'weibo_note'

    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    create_time = db.Column(db.Integer, nullable=False)
    task_id = db.Column(db.Integer, nullable=True)
