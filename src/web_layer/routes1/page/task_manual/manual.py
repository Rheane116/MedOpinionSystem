from flask import Flask, session, render_template, redirect, Blueprint, request, send_file, jsonify, Response, send_from_directory, current_app
from flask import flash, get_flashed_messages
import threading
from src.service_layer.run_task import *
from src.utils.file_utils import *
from src.service_layer.system_service import *

manual_bp = Blueprint('manual', __name__, url_prefix='/page/manual', template_folder='.')

@manual_bp.route('/home')
def manual():
    username = session.get("username")
    user_id = session.get("user_id")
    user_role = session.get("user_role")
    
    tasks = TaskManualService.get_tasks_by_user(user_id)
    tasks_info = []
    for task in tasks:
        tasks_info.append(task.to_dict())

    return render_template("manual.html", username=username, tasks=tasks_info, user_role=user_role)

@manual_bp.route('/create', methods=["POST"])
def manual_create():
    username = session.get("username")
    user_id = session.get("user_id")
    if request.method == "GET":
        return render_template("manual_create.html", username=username)
    else:        
        agency_name=request.form.get('agency_name') 
        other_names=request.form.get('other_names')       
        try:
            TaskManualService.create_task(
                agency_name=agency_name, 
                other_names=other_names, 
                user_id=user_id, 
                gpu=request.form.get('gpu'), 
                batch_size=request.form.get('batch_size'),
                sep=request.form["sep"],
                input_format=request.form["input_format"],
                row = request.form["row"],
                col = request.form["col"],
                file=request.files['upload_data'])
            write_agency_2_json(agency_name, other_names)
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
        return jsonify({'success': True, 'redirect': '/page/manual/home'})   

@manual_bp.route('/update', methods=["GET", "POST"])
def manual_update():
    username = session.get('username')
    user_id = session.get('user_id')
    user_role = session.get("user_role")
    task_id = request.args.get('task_id', type = int)
    if request.method == 'GET':
        task_info = TaskManualService.get_task_by_id(task_id).to_dict()
        return render_template("manual_update.html", username=username, user_role=user_role, task_id=task_info["task_id"], task = task_info)        
    else:  
        agency_name=request.form.get('agency_name') 
        other_names=request.form.get('other_names') 
        try:    
            success = TaskManualService.update_task(
                task_id=request.form.get('task_id'), 
                agency_name=agency_name, 
                other_names=other_names, 
                gpu=request.form.get("gpu"), 
                batch_size=request.form.get("batch_size"), 
                sep=request.form.get("sep"),
                input_format=request.form.get("input_format"),
                row=request.form.get("row"),
                col=request.form.get("col"),
                file=request.files.get("upload_data")
                )
            write_agency_2_json(agency_name, other_names)
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
        return jsonify({'success': True, 'redirect': '/page/manual/home'})

@manual_bp.route('/delete', methods=["POST"])
def manual_delete():
    task_id = request.form.get("task_id")
    from_ = request.form.get("from")
    if from_ == "manage":
        task_id = task_id.split("_")[0]
    #query('delete from task_manual where task_id=%s', (task_id,))
    TaskManualService.delete_task(task_id)
    delete_file(manual_input_folder + task_id + '_.txt')
    delete_file(manual_model_input_folder + task_id + '_.json')
    delete_file(manual_model_output_folder + task_id + '_.ent')
    delete_file(manual_model_output_folder + task_id + '_.rel')
    delete_file(manual_filtered_model_output_folder + task_id + '_.rel')
    delete_file(manual_task_output_excel_folder + task_id + '_.xlsx')

    if from_ == "manual":
        return redirect('/page/manual/home')
    else:
        return redirect('/page/data_manage')

@manual_bp.route('/report')
def manual_report():
    username = session.get("username")
    user_id = session.get("user_id")
    user_role = session.get("user_role")
    task_id = request.args.get('param', type = int)
    task_info = TaskManualService.get_task_by_id(task_id)
 
    output = TaskManualService.get_task_by_id(task_id).result
    return render_template("manual_report.html", username=username, user_role=user_role, task_id=task_id, output=output)

@manual_bp.route('/start', methods=["POST"])
def manual_start():
    task_id = request.form.get("task_id")
    task_info = TaskManualService.get_task_by_id(task_id).to_dict()
    # 在单独的线程中运行爬虫并更新状态
    threading.Thread(
        target=run_task_manual,
        args = (current_app._get_current_object(), task_info,)
    ).start()
    return redirect('/page/manual/home')