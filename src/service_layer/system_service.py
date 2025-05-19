from src.data_access_layer.repositories import *
from config.path import *
import ast

class AuthService:
  """用户认证服务"""
  @staticmethod
  def register(username, password):
    if username == '' or username == None:
      raise ValueError("用户名不能为空")
    if password == '' or password == None:
      raise ValueError("密码不能为空")
    if UserRepository.get_user_by_username(username):
      raise ValueError("该用户名已被注册")
    user = UserRepository.create_user(username, password)
    #user_id = UserRepository.get_user_by_username(username).user_id
    user_id = AuthService.get_user_by_username(username).user_id
    GlobalConfigRepository.create_config(user_id)
    return user
  
  @staticmethod
  def login(username, password):
    user = UserRepository.get_user_by_username(username)
    if user == None:
      raise ValueError("用户未注册，请先注册")
    if user.check_password(password):
      return user
    else:
      raise ValueError("密码错误，请重试")
    
  @staticmethod
  def get_user_by_username(username):
    return UserRepository.get_user_by_username(username)
  
  @staticmethod
  def get_user_by_id(user_id):
    return UserRepository.get_user_by_id(user_id)
  
  @staticmethod
  def update_user_profile(user_id, old_username, new_username, phone, email, agency):
    if old_username != new_username and UserRepository.get_user_by_username(new_username):
      raise ValueError("该用户名已被注册")
    if new_username == None or new_username == '':
      raise ValueError("用户名不能为空")
    if phone == None:
      phone = ""
    if email == None:
      email = ""
    if agency == None:
      agency = ""
    return UserRepository.update_user_profile(user_id, new_username, phone, email, agency)
      
class TaskService:
  """任务管理服务"""

  @staticmethod
  def get_task_by_id(task_id):
    return TaskRepository.get_task_by_id(task_id)

  @staticmethod
  def get_tasks_by_user(user_id):
    return TaskRepository.get_tasks_by_user(user_id)
  
  @staticmethod
  def create_task(agency_name, other_names, platform, user_id, start_time, end_time, max_note, max_comment, if_level, gpu, batch_size):
    if agency_name == None or agency_name == "":
      raise ValueError("请填写医疗机构全称")
    if platform == None or platform == []:
      raise ValueError("请选择目标平台")
    platform = '///'.join(platform)
    try:
      if max_note == None or max_note == '':
        max_note = 300
      max_note = int(max_note)
      if max_note <= 0 or max_note > 1000:
        raise ValueError("最大爬取帖数需为1-1000之间的整数")
    except Exception as e:
      raise ValueError("最大爬取帖数需为1-1000之间的整数")
    try:
      if max_comment == None or max_comment == '':
        max_comment = 100
      max_comment = int(max_comment)
      if max_comment <= 0 or max_comment > 500:
        raise ValueError("每个帖子最大爬取评论数需为1-500之间的整数")
    except Exception as e:
      raise ValueError("每个帖子最大爬取评论数需为1-500之间的整数")
    
    if if_level == '是':
      if_level= 1
    else:
      if_level = 0
    if gpu is None or gpu=='':
      gpu = '0'

    try:
      if batch_size == None or batch_size == '':
        batch_size = 8
      batch_size = int(batch_size)
      if batch_size <= 0:
        raise ValueError("推理批量大小需为大于0的整数")
    except Exception as e:
      raise ValueError("推理批量大小需为大于0的整数")
    
    time_tuple = time.localtime(time.time())
    #create_time = str(time_tuple[0]) + '-' + str(time_tuple[1]) + '-' + str(time_tuple[2])+ '-' + str(time_tuple[3])+ '-' + str(time_tuple[4])+ '-' + str(time_tuple[5])
    create_time = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(time_tuple[0], time_tuple[1], time_tuple[2],time_tuple[3],time_tuple[4],time_tuple[5])
    if start_time == None or start_time == '':
      start_time = "1970-01-01"
    if end_time == None or end_time == '':
      end_time = "{:04d}-{:02d}-{:02d}".format(time_tuple[0], time_tuple[1], time_tuple[2])
    if other_names is None:
      other_names = ''  
    return TaskRepository.create_task(agency_name, other_names, platform, user_id, start_time, end_time, create_time, max_note, max_comment, if_level, gpu, batch_size)  
  
  @staticmethod
  def update_task(task_id, agency_name, other_names, platform, start_time, end_time, max_note, max_comment, if_level, gpu, batch_size):
    if agency_name == None or agency_name == "":
      raise ValueError("请填写医疗机构全称")
    if platform == None or platform == []:
      raise ValueError("请选择目标平台")
    platform = '///'.join(platform)
    try:
      if max_note == None or max_note == '':
        max_note = 300
      max_note = int(max_note)
      if max_note <= 0 or max_note > 1000:
        raise ValueError("最大爬取帖数需为1-1000之间的整数")
    except Exception as e:
      raise ValueError("最大爬取帖数需为1-1000之间的整数")
    try:
      if max_comment == None or max_comment == '':
        max_comment = 100
      max_comment = int(max_comment)
      if max_comment <= 0 or max_comment > 500:
        raise ValueError("每个帖子最大爬取评论数需为1-500之间的整数")
    except Exception as e:
      raise ValueError("每个帖子最大爬取评论数需为1-500之间的整数")
    
    if if_level == '是':
      if_level= 1
    else:
      if_level = 0
    if gpu is None or gpu=='':
      gpu = '0'

    try:
      if batch_size == None or batch_size == '':
        batch_size = 8
      batch_size = int(batch_size)
      if batch_size <= 0:
        raise ValueError("推理批量大小需为大于0的整数")
    except Exception as e:
      raise ValueError("推理批量大小需为大于0的整数")

    time_tuple = time.localtime(time.time())
    create_time = str(time_tuple[0]) + '-' + str(time_tuple[1]) + '-' + str(time_tuple[2])+ '-' + str(time_tuple[3])+ '-' + str(time_tuple[4])+ '-' + str(time_tuple[5])
    if start_time == None or start_time == '':
      start_time = "1970-01-01"
    if end_time == None or end_time == '':
      end_time = str(time_tuple[0]) + '-' + str(time_tuple[1]) + '-' + str(time_tuple[2])
    if other_names is None:
      other_names = ''

    return TaskRepository.update_task(task_id, agency_name, other_names, platform, start_time, end_time, create_time, max_note, max_comment, if_level, gpu, batch_size)
  
  @staticmethod
  def delete_task(task_id):
    return TaskRepository.delete_task(task_id)
  
  @staticmethod
  def update_task_result(task_id, result):
    return TaskRepository.update_task_result(task_id, result)
  
  @staticmethod
  def update_task_status(task_id, status):
    return TaskRepository.update_task_status(task_id, status)

    
  @staticmethod
  def save_cookies(cookies, platform, username):
    TaskRepository.save_cookies(cookies, platform, username)


class TaskManualService:
  """手动任务管理服务"""
  @staticmethod
  def get_task_by_id(task_id):
    return TaskManualRepository.get_task_by_id(task_id)
  
  @staticmethod
  def get_tasks_by_user(user_id):
    return TaskManualRepository.get_tasks_by_user(user_id)
  
  @staticmethod
  def create_task(agency_name, other_names, user_id, gpu, batch_size, sep, input_format, row, col, file):
    try:
        if row == None or row == '':
            row = 2
        row = int(row)
        if row <= 0:
            raise ValueError("文本开始行数需为大于0的整数")
    except Exception as e:
        raise ValueError("文本开始行数需为大于0的整数")
    try:
        if col == None or col == '':
            col = 4
        col = int(col)
        if col <= 0:
            raise ValueError("文本所在列数需为大于0的整数")
    except Exception as e:
        raise ValueError("文本所在列数需为大于0的整数")
    if input_format == "txt" and not file.filename.endswith('.txt'):
      raise ValueError("请导入txt文件")
    if input_format == "xlsx" and not file.filename.endswith('.xlsx'):
      raise ValueError("请导入xlsx文件")
    if gpu is None or gpu=='':
      gpu = '0'
    try:
      if batch_size == None or batch_size == '':
        batch_size = 8
      batch_size = int(batch_size)
      if batch_size <= 0:
        raise ValueError("推理批量大小需为大于0的整数")
    except Exception as e:
      raise ValueError("推理批量大小需为大于0的整数")
    time_tuple = time.localtime(time.time())
    create_time = str(time_tuple[0]) + '-' + str(time_tuple[1]) + '-' + str(time_tuple[2])+ '-' + str(time_tuple[3])+ '-' + str(time_tuple[4])+ '-' + str(time_tuple[5])
    if other_names is None:
      other_names = '' 

    task_m_id = TaskManualRepository.create_task(agency_name, other_names, user_id, create_time, gpu, batch_size, sep, input_format, row, col).task_id
    if input_format == "xlsx":
        excel_2_txt(file, task_m_id, row, col)
    else:
        file.save(f'{manual_input_folder}{task_m_id}_.txt')
  
  @staticmethod
  def update_task(task_id, agency_name, other_names, gpu, batch_size, sep, input_format, row, col, file):
    try:
        if row == None or row == '':
            row = 2
        row = int(row)
        if row <= 0:
            raise ValueError("文本开始行数需为大于0的整数")
    except Exception as e:
        raise ValueError("文本开始行数需为大于0的整数")
    try:
        if col == None or col == '':
            col = 4
        col = int(col)
        if col <= 0:
            raise ValueError("文本所在列数需为大于0的整数")
    except Exception as e:
        raise ValueError("文本所在列数需为大于0的整数")
    if input_format == "txt" and not file.filename.endswith('.txt'):
      raise ValueError("请导入txt文件")
    if input_format == "xlsx" and not file.filename.endswith('.xlsx'):
      raise ValueError("请导入xlsx文件")
    if gpu is None or gpu=='':
      gpu = '0'
    if batch_size is None or batch_size == '' or batch_size <= 0 or not isinstance(batch_size, int):
      batch_size = 8
    time_tuple = time.localtime(time.time())
    create_time = str(time_tuple[0]) + '-' + str(time_tuple[1]) + '-' + str(time_tuple[2])+ '-' + str(time_tuple[3])+ '-' + str(time_tuple[4])+ '-' + str(time_tuple[5])
    if other_names is None:
      other_names = '' 
    TaskManualRepository.update_task(task_id, agency_name, other_names, create_time, gpu, batch_size, sep, input_format, row, col)
    if input_format == "xlsx":
        excel_2_txt(file, task_id, row, col)
    else:
        file.save(f'{manual_input_folder}{task_id}_.txt')
  
  @staticmethod
  def delete_task(task_id):
    return TaskManualRepository.delete_task(task_id)
  
class TaskDataService:
  @staticmethod
  def update_data_output(data_id, rels_list_add):
    rels_list_str = TaskDataRepository.get_data_by_id(data_id).output
    if rels_list_str == None or rels_list_str:
      success = TaskDataRepository.update_data_output(data_id, str(rels_list_add))
    else:
      rels_list = ast.literal_eval(rels_list_str)
      rels_list = list(set(rels_list + rels_list_add))
      success = TaskDataRepository.update_data_output(data_id, str(rels_list))
    return success

class GlobalConfigService:
  """全局任务配置服务"""
  
  @staticmethod
  def get_config(user_id):
    config = GlobalConfigRepository.get_config_by_userid(user_id)
    config.agency_name = config.agency_name if config.agency_name is not None else ""
    config.other_names = config.other_names if config.other_names is not None else ""
    config.max_note = config.max_note if config.max_note is not None else ""
    config.max_comment = config.max_comment if config.max_comment is not None else ""
    config.gpu = config.gpu if config.gpu is not None else ""
    config.batch_size = config.batch_size if config.batch_size is not None else ""
    return config
  
  @staticmethod
  def update_config(user_id, agency_name, other_names, max_note, max_comment, if_level, gpu, batch_size):
    try:
      if max_note == None or max_note == '':
        max_note = 300
      max_note = int(max_note)
      if max_note <= 0 or max_note > 1000:
        raise ValueError("最大爬取帖数需为1-1000之间的整数")
    except Exception as e:
      raise ValueError("最大爬取帖数需为1-1000之间的整数")
    try:
      if max_comment == None or max_comment == '':
        max_comment = 100
      max_comment = int(max_comment)
      if max_comment <= 0 or max_comment > 500:
        raise ValueError("每个帖子最大爬取评论数需为1-500之间的整数")
    except Exception as e:
      raise ValueError("每个帖子最大爬取评论数需为1-500之间的整数")
    
    if if_level == '是':
      if_level= 1
    else:
      if_level = 0
    if gpu is None or gpu=='':
      gpu = '0'

    try:
      if batch_size == None or batch_size == '':
        batch_size = 8
      batch_size = int(batch_size)
      if batch_size <= 0:
        raise ValueError("推理批量大小需为大于0的整数")
    except Exception as e:
      raise ValueError("推理批量大小需为大于0的整数")
    
    if agency_name == None:
      agency_name = ""
    if other_names == None:
      other_names = ""
    return GlobalConfigRepository.update_config(user_id, agency_name, other_names, max_note, max_comment, if_level, gpu, batch_size)


