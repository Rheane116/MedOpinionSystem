from flask import Flask, session, render_template, redirect, Blueprint, request, send_file, jsonify, Response, send_from_directory, current_app
import asyncio
import time
import json
import os
import threading
from src.service_layer.system_service import *
from src.data_access_layer.repositories import CrawlDataRepository
from src.service_layer.run_task import *
from src.utils.file_utils import *

task_bp = Blueprint('task', __name__, url_prefix='/page/task', template_folder='.')

@task_bp.route('/home')
def task():
    username = session.get('username')
    user_id = session.get('user_id')
    user_role = session.get('user_role')
    tasks = TaskService.get_tasks_by_user(user_id)
    tasks_info = []
    for task in tasks:
        tasks_info.append(task.to_dict())
    # 按任务编号降序排列，确保页面“任务列表”默认按任务编号从大到小展示
    try:
        tasks_info = sorted(tasks_info, key=lambda x: int(x['task_id']), reverse=True)
    except (ValueError, TypeError, KeyError):
        # 如果 task_id 不是纯数字或缺失，则回退为字符串降序排序
        tasks_info = sorted(tasks_info, key=lambda x: str(x.get('task_id', '')), reverse=True)
    return render_template("task.html", username=username, user_role=user_role, tasks=tasks_info)

@task_bp.route('/create', methods=["POST"])
def task_create():
    username = session.get('username')
    user_id = session.get('user_id')
    user_role = session.get('user_role')
    if request.method == "GET":
        return render_template("task_create.html", username=username, user_role=user_role)
    else:
        cookies = {
            '知乎' : request.form.get("cookies_zhihu"),
            '哔哩哔哩' : request.form.get("cookies_bili"),
            '微博' : request.form.get("cookies_wb"),
            '小红书' : request.form.get("cookies_xhs"),
            '抖音' : request.form.get("cookies_dy")
        }
        agency_name = request.form.get("agency_name")
        other_names = request.form.get("other_names")
        platform = request.form.getlist("platform_name")
        try:
            task_ = TaskService.create_task(
                agency_name=agency_name, 
                other_names=other_names, 
                platform=platform, 
                user_id=user_id, 
                start_time=request.form.get("start_time"), 
                end_time=request.form.get("end_time"), 
                max_note=request.form.get("max_note"), 
                max_comment=request.form.get("max_comment"), 
                if_level=request.form.get("if_level"), 
                gpu=request.form.get("gpu"), 
                batch_size=request.form.get("batch_size"))
            TaskService.save_cookies(cookies, platform, username)
            write_agency_2_json(agency_name, other_names)
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
        return jsonify({'success': True, 'redirect': '/page/task/home'})


@task_bp.route('/update', methods=['GET', 'POST'])
def task_update():
    username = session.get('username')
    user_id = session.get('user_id')
    user_role = session.get('user_role')
    if request.method == 'GET':
        task_id = request.args.get('task_id', type = int)
        task_info = TaskService.get_task_by_id(task_id).to_dict()
        return render_template("task_update.html", username=username, user_role=user_role, task=task_info)
            
    else:
        agency_name = request.form.get("agency_name")
        other_names = request.form.get("other_names")
        platform = request.form.getlist("platform_name")
        cookies = {
            '知乎' : request.form.get("cookies_zhihu"),
            '哔哩哔哩' : request.form.get("cookies_bili"),
            '微博' : request.form.get("cookies_wb"),
            '小红书' : request.form.get("cookies_xhs"),
            '抖音' : request.form.get("cookies_dy"),
        }
        try:
            success = TaskService.update_task(
                task_id=request.form.get('task_id'), 
                agency_name=agency_name, 
                other_names=other_names, 
                platform=platform, 
                start_time=request.form.get("start_time"), 
                end_time=request.form.get("end_time"), 
                max_note=request.form.get("max_note"), 
                max_comment=request.form.get("max_comment"), 
                if_level=request.form.get("if_level"), 
                gpu=request.form.get("gpu"), 
                batch_size=request.form.get("batch_size"))
            TaskService.save_cookies(cookies, platform, username) 
            write_agency_2_json(agency_name, other_names)
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})    
        return jsonify({'success': True, 'redirect': '/page/task/home'})


@task_bp.route('/delete', methods=["POST"])
def task_delete():
    task_id = request.form.get("task_id")
    TaskDataRepository.delete_data_by_task(task_id)
    TaskService.delete_task(task_id)
    delete_file(model_input_folder + task_id + '_.json')
    delete_file(model_output_folder + task_id + '_.ent')
    delete_file(model_output_folder + task_id + '_.rel')
    delete_file(filtered_model_output_folder + task_id + '_.rel')
    delete_file(task_output_excel_folder + task_id + '_.xlsx')
    CrawlDataRepository.delete_raw_data(task_id)
    if request.form.get("from") == "crawl":
        return redirect("/page/task/home")
    else:
        return redirect("/page/data_manage")

@task_bp.route('/delete_raw', methods=["POST"])
def task_delete_raw():
    task_id = request.form.get("task_id1")
    CrawlDataRepository.delete_raw_data(task_id)
    return redirect("/page/data_manage")

@task_bp.route('/report')
def task_report():
    username = session.get("username")
    user_id = session.get("user_id")
    user_role = session.get("user_role")
    task_id = request.args.get('param', type = int)
    task_info = TaskService.get_task_by_id(task_id)
 
    output = TaskService.get_task_by_id(task_id).result
    return render_template("task_report.html", username=username, user_role=user_role, task_id=task_id, output = output)


@task_bp.route('/task_start', methods=["POST"])
def task_start():
    task_id = request.form.get("task_id")
    task_info = TaskService.get_task_by_id(task_id).to_dict()
    threading.Thread(
        target=run_task,
        args = (current_app._get_current_object(), task_info,)
    ).start()
    return redirect('/page/task/home')
    

