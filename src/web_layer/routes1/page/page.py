from flask import Flask, session, render_template, redirect, Blueprint, request, send_file, jsonify, Response, send_from_directory, current_app

from config.path import *


from src.web_layer.routes1.page.task import task
from src.web_layer.routes1.page.task_manual import manual
from src.web_layer.routes1.page.deepseek import deepseek

from src.service_layer.system_service import AuthService,GlobalConfigService
from src.data_access_layer.repositories import TaskRepository, TaskManualRepository, TaskDataRepository, CrawlDataRepository
from src.utils.file_utils import *

page_bp = Blueprint('page', __name__, url_prefix='/page', template_folder='.')
page_bp.register_blueprint(task.task_bp, url_prefix='/task')
page_bp.register_blueprint(manual.manual_bp, url_prefix='/manual')
page_bp.register_blueprint(deepseek.deepseek_bp, url_prefix='/deepseek')



@page_bp.route('/home')
def home():
    username = session.get('username')
    user_role = session.get('user_role')
    return render_template('index.html', username=username, user_role=user_role)


@page_bp.route('/get_task_data', methods=["GET"])
def get_task_data():
    task_id = request.args.get('task_id', type=str)
    datalist = TaskDataRepository.get_data_by_task(task_id)
    taskdata_output_2_excel(task_id, datalist)
    return send_file(f"/disk2/zyy/medOpinionSystem/shared/task/task_output_excel/{task_id}_.xlsx", as_attachment=True)

@page_bp.route('/get_task_data_manual', methods=["GET"])
def get_task_data_manual():
    task_id = request.args.get('task_id', type=str)
    return send_file(f"/disk2/zyy/medOpinionSystem/shared/manual/task_output_excel/{task_id}_.xlsx", as_attachment=True)

@page_bp.route('/user_profile', methods=["GET", "POST"])
def user_profile():
    username = session.get("username")
    user_id = session.get("user_id")
    user_role = session.get('user_role')
    if request.method == "GET":
        user = AuthService.get_user_by_id(user_id)
        user.phone = ""if user.phone is None else user.phone
        user.email = "" if user.email is None else user.email
        user.agency = "" if user.agency is None else user.agency
        return render_template("user_profile.html", username=username, user=user, user_role=user_role)
    else:
        try:
            AuthService.update_user_profile(user_id, username, request.form["username"], request.form["phone"], request.form["email"], request.form["agency"])
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
        session["username"] = request.form["username"]
        return jsonify({'success': True, 'redirect': '/page/user_profile'})

@page_bp.route('/global_config', methods=["GET", "POST"])
def global_config():
    username = session.get('username')
    user_id = session.get('user_id')
    user_role = session.get('user_role')
    if request.method == "GET":
        config = GlobalConfigService.get_config(user_id)
        agency = config.agency_name
        agency_list = read_json(AGENCY_LIST_PATH)
        if agency in agency_list.keys():
            other_names = '///'.join(agency_list[agency])
        else:
            other_names = config.other_names
        return render_template("global_config.html", username=username, user_role=user_role,
        other_names=other_names, config=config)
    else:
        try:
            GlobalConfigService.update_config(user_id, request.form["agency_name"],request.form["other_names"], request.form["max_note"], request.form["max_comment"], request.form["if_level"], request.form["gpu"], request.form["bsz"])
            write_agency_2_json(request.form["agency_name"], request.form["other_names"])
            return jsonify({'success': True, 'redirect':'/page/global_config'})
        except Exception as e:
            return jsonify({"success":False, "message":str(e)})

    
@page_bp.route('/global_config_import', methods=["GET"])
def global_config_import():
    username = session.get('username')
    user_id = session.get('user_id')
    try:
        config = GlobalConfigService.get_config(user_id).to_dict()
        return jsonify({'success':True, 'config':config})
    except Exception as e:
        return jsonify({'success':False, 'message':str(e)})
    
@page_bp.route('/data_manage', methods=["GET"])
def data_manage():
    username = session.get("username")
    user_id = session.get("user_id")
    user_role = session.get("user_role")

    tasks = TaskRepository.get_all_tasks()
    tasks_m = TaskManualRepository.get_all_tasks()
    task_list = []
    for task in tasks:
        data_num = TaskDataRepository.get_count_by_taskid(task.task_id)
        raw_data_num= CrawlDataRepository.get_raw_data_count(task.task_id)
        task_list.append({"task_id":task.task_id, "user_id": task.user_id,"agency_name":task.agency_name,  "platform": task.platform, "create_time": task.create_time, "status": task.status, "data_num": data_num, "raw_data_num": raw_data_num})
    for task_m in tasks_m:
        data_num_m = TaskManualRepository.get_count_by_taskid(task_m.task_id)
        task_list.append({"task_id":str(task_m.task_id)+"_manual", "user_id": task_m.user_id,"agency_name":task_m.agency_name,  "platform": "-", "create_time": task_m.create_time, "status": task_m.status, "data_num": data_num_m, "raw_data_num": 0})
    return render_template("data_manage.html", username=username, user_role=user_role, tasks=task_list)


        

    
  
















