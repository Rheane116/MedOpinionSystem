from .database import db
from .models import *
import time
from config.path import *
from src.utils.file_utils import *


class UserRepository:
  """用户数据仓库"""

  @staticmethod
  def get_user_by_id(user_id):
    return User.query.get(user_id)

  @staticmethod
  def get_user_by_username(username):
    return User.query.filter_by(username=username).first()

  @staticmethod
  def create_user(username, pwd):
    time_tuple = time.localtime(time.time())
    create_time =  str(time_tuple[0]) + '-' + str(time_tuple[1]) + '-' + str(time_tuple[2])+ '-' + str(time_tuple[3])+ '-' + str(time_tuple[4])+ '-' + str(time_tuple[5])
    user = User(username=username, create_time=create_time, role=0)
    user.set_password(pwd)
    db.session.add(user)
    db.session.commit()
    return user
  
  @staticmethod
  def update_user_profile(user_id, username, phone, email, agency):
    user = User.query.get(user_id)
    if user:
      user.username = username
      user.phone = phone
      user.email = email
      user.agency = agency
      db.session.commit()
      return True
    return False

class TaskRepository:
  """任务数据仓库"""
  @staticmethod
  def get_all_tasks():
    return Task.query.all()

  @staticmethod
  def get_task_by_id(task_id):
     return Task.query.filter_by(task_id=task_id).first()
  
  @staticmethod
  def get_tasks_by_user(user_id):
     return Task.query.filter_by(user_id=user_id).all()

  @staticmethod
  def create_task(agency_name, other_names, platform, user_id, start_time, end_time, create_time, max_note, max_comment, if_level, gpu, batch_size):
    task = Task(agency_name=agency_name, other_names=other_names,platform=platform, status=0, user_id=user_id, start_time=start_time, end_time=end_time, create_time=create_time, max_note=max_note, max_comment=max_comment, if_level=if_level, gpu=gpu, batch_size=batch_size)
    db.session.add(task)
    db.session.commit()
    return task
  
  @staticmethod
  def update_task(task_id, agency_name, other_names, platform, start_time, end_time, create_time, max_note, max_comment, if_level, gpu, batch_size):
    task = Task.query.filter_by(task_id=task_id).first()
    if task:
      task.agency_name = agency_name
      task.other_names=other_names
      task.platform = platform
      task.start_time=start_time
      task.end_time = end_time
      task.create_time = create_time
      task.max_note = max_note
      task.max_comment = max_comment
      task.if_level = if_level
      task.gpu = gpu
      task.batch_size = batch_size
      db.session.commit()
      return True
    return False
  
  @staticmethod
  def delete_task(task_id):
    try:
      deleted_count = Task.query.filter_by(task_id=task_id).delete()
      db.session.commit()
      return deleted_count > 0
    except Exception as e:
      db.session.rollback()
      return False
     
  @staticmethod
  def update_task_result(task_id, result):
     task = Task.query.filter_by(task_id=task_id).first()
     if task:
        task.result = result
        db.session.commit()
        return True
     return False
  
  @staticmethod
  def update_task_status(task_id, status):
    task = Task.query.filter_by(task_id=task_id).first()
    if task:
      task.status = status
      db.session.commit()
      return True
    return False

  @staticmethod
  def save_cookies(cookies, platforms, username):
    for platform in platforms:
        if platform not in cookies:
           continue
        if cookies[platform] == None or cookies[platform] == "":
          continue
        exists = CookieAccount.query.filter_by(account_name=username, platform_name=platform_map.get(platform)).first()
        if exists:
           if exists.status == -1 and exists.cookies == cookies[platform]:
              raise Exception(f"平台'{platform}'用户'{exists.account_name}'的cookies已经失效，请重新提取")
              return
           exists.cookies = cookies[platform]
           exists.status = 0
           db.session.commit()
        else:
           account = CookieAccount(
              account_name=username,
              platform_name=platform_map.get(platform),
              cookies=cookies[platform]
           )
           db.session.add(account)
           db.session.commit()


class TaskManualRepository:
  """手动任务数据仓库"""

  @staticmethod
  def get_all_tasks():
    return TaskManual.query.all()

  @staticmethod
  def get_task_by_id(task_id):
    return TaskManual.query.filter_by(task_id=task_id).first()

  @staticmethod
  def get_tasks_by_user(user_id):
    return TaskManual.query.filter_by(user_id=user_id).all()
  
  @staticmethod
  def get_count_by_taskid(task_id):
    return len(read_json(f"{manual_model_input_folder}{task_id}_.json"))

  @staticmethod
  def create_task(agency_name, other_names, user_id, create_time, gpu, batch_size, sep, input_format, row, col):
    task_m = TaskManual(agency_name=agency_name, other_names=other_names,status=0, user_id=user_id, create_time=create_time, gpu=gpu, batch_size=batch_size, sep=sep, input_format=input_format, row=row, col=col)
    db.session.add(task_m)
    db.session.commit()
    return task_m

  @staticmethod
  def update_task(task_id, agency_name, other_names, create_time, gpu, batch_size, sep, input_format, row, col):
    task_m = TaskManual.query.filter_by(task_id=task_id).first()
    if task_m:
      task_m.agency_name = agency_name
      task_m.other_names=other_names
      task_m.create_time = create_time
      task_m.gpu = gpu
      task_m.batch_size = batch_size
      task_m.sep = sep
      task_m.input_format = input_format
      task_m.row = row
      task_m.col = col
      db.session.commit()
      return True
    return False

  @staticmethod
  def delete_task(task_id):
    try:
      deleted_count = TaskManual.query.filter_by(task_id=task_id).delete()
      db.session.commit()
      return deleted_count > 0
    except Exception as e:
      db.session.rollback()
      return False

  @staticmethod
  def update_taskManual_status(task_id, status):
    task_m = TaskManual.query.filter_by(task_id=task_id).first()
    if task_m:
      task_m.status = status
      db.session.commit()
      return True
    return False
      
  @staticmethod
  def update_taskManual_result(task_id, result):
    task_m = TaskManual.query.filter_by(task_id=task_id).first()
    if task_m:
      task_m.result = result
      db.session.commit()
      return True
    return False


class TaskDataRepository:
  """任务数据的数据仓库"""
  @staticmethod
  def get_data_by_id(data_id):
    return TaskData.query.get(data_id)

  @staticmethod
  def get_data_by_task(task_id):
    return TaskData.query.filter_by(task_id=task_id).all()
  
  @staticmethod
  def create_taskdata(data_id, task_id, platform, time, text):
    task_data = TaskData(data_id=data_id, task_id=task_id, platform=platform, time=time, text=text)
    db.session.add(task_data)
    db.session.commit()
    return task_data


  @staticmethod
  def update_data_output(data_id, output):
    taskData = TaskData.query.filter_by(data_id=data_id).first()
    if taskData:
        taskData.output = output
        db.session.commit()
        return True
    return False
  
  @staticmethod
  def delete_data_by_task(task_id):
    try:
      deleted_count = TaskData.query.filter_by(task_id=task_id).delete()
      db.session.commit()
      return deleted_count > 0
    except Exception as e:
      db.session.rollback()
      return False
    
  @staticmethod
  def get_count_by_taskid(task_id):
    return TaskData.query.filter_by(task_id=task_id).count()


class GlobalConfigRepository:
  """全局配置数据仓库"""
  @staticmethod
  def get_config_by_userid(user_id):
    return GlobalConfig.query.get(user_id)
  
  @staticmethod
  def create_config(user_id):
    config = GlobalConfig(user_id=user_id)
    db.session.add(config)
    db.session.commit()
    return config
  
  @staticmethod
  def update_config(user_id, agency_name, other_names, max_note, max_comment, if_level, gpu, batch_size):
    config = GlobalConfig.query.filter_by(user_id=user_id).first()
    if config:
      config.agency_name = agency_name
      config.other_names = other_names
      config.max_note = max_note
      config.max_comment = max_comment
      config.if_level = if_level
      config.gpu = gpu
      config.batch_size = batch_size
      db.session.commit()
      return True
    return False
  
platform_2_model = {
  '知乎': [ZhihuComment, ZhihuContent],
  '小红书': [XhsComment, XhsContent],
  '哔哩哔哩': [BilibiliComment, BilibiliContent],
  '微博': [WeiboComment, WeiboContent]
}

class CrawlDataRepository:
  """原始爬虫数据的数据仓库"""
  @staticmethod
  def get_comments(platform, task_id):
    return platform_2_model.get(platform)[0].query.filter_by(task_id=task_id).all()
  
  @staticmethod
  def get_contents(platform, task_id):
    return platform_2_model.get(platform)[1].query.filter_by(task_id=task_id).all()
  
  @staticmethod
  def get_comments_count(platform, task_id):
    return platform_2_model.get(platform)[0].query.filter_by(task_id=task_id).count()
  
  @staticmethod
  def get_contents_count(platform, task_id):
    return platform_2_model.get(platform)[1].query.filter_by(task_id=task_id).count()
  
  @staticmethod
  def delete_comments(platform, task_id):
    platform_2_model.get(platform)[0].query.filter_by(task_id=task_id).delete()
    db.session.commit()

  @staticmethod
  def delete_contents(platform, task_id):
    platform_2_model.get(platform)[1].query.filter_by(task_id=task_id).delete()
    db.session.commit()

  @staticmethod
  def get_raw_data_count(task_id):
    return CrawlDataRepository.get_comments_count("知乎", task_id) + CrawlDataRepository.get_comments_count("小红书", task_id) + CrawlDataRepository.get_comments_count("哔哩哔哩", task_id) + CrawlDataRepository.get_comments_count("微博", task_id) + CrawlDataRepository.get_contents_count("知乎", task_id) + CrawlDataRepository.get_contents_count("小红书", task_id) + CrawlDataRepository.get_contents_count("哔哩哔哩", task_id) + CrawlDataRepository.get_contents_count("微博", task_id)
  
  @staticmethod
  def delete_raw_data(task_id):
    CrawlDataRepository.delete_comments("知乎", task_id)
    CrawlDataRepository.delete_comments("小红书", task_id)
    CrawlDataRepository.delete_comments("哔哩哔哩", task_id)
    CrawlDataRepository.delete_comments("微博", task_id)
    CrawlDataRepository.delete_contents("知乎", task_id)
    CrawlDataRepository.delete_contents("小红书", task_id)
    CrawlDataRepository.delete_contents("哔哩哔哩", task_id)
    CrawlDataRepository.delete_contents("微博", task_id)
   
     

      
  
