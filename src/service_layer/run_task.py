import subprocess
from config.path import *
from src.data_access_layer.repositories import TaskRepository,TaskManualRepository, UserRepository
from src.service_layer.data_process import *

def run_task(app, info):
    print(f"run_task:任务参数为：")
    print(info)

    with app.app_context():
        task_id = info['task_id']
        platforms = info['platform'].split('///')
        status = info['status'] 
        username = UserRepository.get_user_by_id(info["user_id"]).username
        for i in range(len(platforms)):
            platforms[i] = platform_map.get(platforms[i])

        if status == 0 or status == 4:
            TaskRepository.update_task_status(task_id, 1)
            status = 1
        if status == 1:
            processes = []
            success = False
            for i, platform in enumerate(platforms):
                process = subprocess.Popen([crawler_path, platforms[i], info['agency_name'], 
                                            str(info['max_note']), str(info['max_comment']), str(info['if_level']), str(info['task_id']), username], shell=False, text=True)
                processes.append(process)
                print(f"run_task:启动爬虫子进程，平台：{platform},进程ID：{process.pid}")
                print("传入的参数为:")
                print(info)
            for i, platform in enumerate(platforms):
                processes[i].wait()
                print(f"run_task:{platform}平台爬虫结束，进程退出码为{processes[i].returncode}")
                if processes[i].returncode != 0:
                    print(f"run_task:{platform}平台爬虫失败，进程退出码为{processes[i].returncode}")
                else:
                    success = True
                    print(f"run_task:{platform}平台爬虫完成")
            if not success:
                TaskRepository.update_task_status(task_id, 4)
                return
            print(f"开始处理爬虫数据")
            num = crawl_process(info, app)
            # 没有爬取到有效数据
            if num == 0:
                print(f"run_task:任务{task_id}没有爬取到有效数据")
                TaskRepository.update_task_status(task_id, 4)
                return
            print(f"爬虫数据处理完成")
        TaskRepository.update_task_status(task_id, 2)
        print(f"run_task:传入bispn的task_id为{task_id}")
        process = subprocess.Popen([bispn_path, str(task_id), info['gpu'], str(info['batch_size'])],
                                    shell=False, text=True)
        process.wait()
        if process.returncode != 0:
            print(f"run_task: 情感分析失败，进程退出码为{process.returncode}")
            TaskRepository.update_task_status(task_id, 5)
            return    
        output_process(info, app)
        TaskRepository.update_task_status(task_id, 3)
    return


def run_task_manual(app, info):
    print(f"run_task:任务参数为：")
    print(info)     

    with app.app_context():
        task_id = info["task_id"]

        TaskManualRepository.update_taskManual_status(task_id, 2)
        num = manual_input_process(info)
        if num == 0:
            print(f"run_task_manual:任务{task_id}无有效的输入数据")
            TaskManualRepository.update_taskManual_status(task_id, 5)
            return
        print(f"run_task_manual:传入bispn的task_id为{task_id}")
        process = subprocess.Popen([bispn_m_path, str(task_id), info["gpu"], str(info["batch_size"])], shell=False, text=True)
        process.wait()
        if process.returncode != 0:
            print(f"run_task_manual: 情感分析失败，进程退出码为{process.returncode}")
            TaskManualRepository.update_task_status(task_id, 5)
            return
        output_process(info, app, manual=True)
        TaskManualRepository.update_taskManual_status(task_id, 3)
        return
