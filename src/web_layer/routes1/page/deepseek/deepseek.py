from flask import Flask, session, render_template, redirect, Blueprint, request, send_file, jsonify, Response, send_from_directory
from flask import flash, get_flashed_messages
import time
import json
from openai import OpenAI
from config.deepseek_config import *
from src.data_access_layer.repositories import TaskDataRepository


deepseek_bp = Blueprint('deepseek', __name__, url_prefix='/page/deepseek', template_folder='.')

prompt_new = er_prompt
msg = ''

@deepseek_bp.route('/home', methods=["GET"])
def deepseek():
    global msg
    msg_tmp = msg

    username = session.get('username')
    user_role = session.get('user_role')

    print(f'---------msg:{msg}')
    if msg != '':
        msg = ''
    
    return render_template('deepseek.html', er_prompt=prompt_new, msg = msg_tmp, username=username, user_role=user_role)


client = OpenAI(
    api_key=API_CONFIG["api_key"],
    base_url=API_CONFIG["base_url"]
)

messages = [{"role": "system", "content": "你是一个严谨的AI助手"}]

@deepseek_bp.route('/send', methods=['POST'])
def send():
    try:
        user_message = request.json['message']
        model = request.json.get('model', 'deepseek-chat')  # Default to reasoner if not specified
        print(f"\n用户问题: {user_message}")
        print(f"使用模型: {model}")
        def generate():
            global messages
            full_response = ""
            for attempt in range(3):  # 最多尝试3次
                try:
                    print(f"\n第{attempt + 1}次尝试获取回答")

                    if len(messages) >= 20:
                        messages = [{"role": "system", "content": "你是一个严谨的AI助手"}]
                    messages.append({"role": "user", "content": user_message})
                    response = client.chat.completions.create(
                        #model='deepseek-reasoner',
                        model=model,
                        messages=messages,
                        stream=True,
                        temperature=0.7,  # 添加温度参数
                        max_tokens=2000   # 限制最大token数
                    )
                    print('-----------messages:')
                    print(messages)
                    has_content = False
                    for chunk in response:
                        if chunk.choices[0].delta.content is not None:
                            content = chunk.choices[0].delta.content
                            full_response += content
                            has_content = True
                            print(content, end='', flush=True)
                            yield f"data: {json.dumps({'content': content})}\n\n"
                    
                    if has_content and full_response.strip():  # 如果响应不为空，则返回
                        #print(f"\n完整回答: {full_response}")
                        messages.append({"role":"assistant", "content":full_response})
                        return
                    else:
                        print(f"\n第{attempt + 1}次尝试返回为空")
                        messages = messages[:-1]  # 移除用户消息
                        if attempt < 2:  # 如果不是最后一次尝试，等待一段时间后重试
                            time.sleep(1)  # 等待1秒后重试
                            continue
                        
                except Exception as e:
                    print(f"\n第{attempt + 1}次尝试发生错误: {str(e)}")
                    messages = messages[:-1]  # 移除用户消息
                    if attempt < 2:  # 如果不是最后一次尝试，等待一段时间后重试
                        time.sleep(1)  # 等待1秒后重试
                        continue
                    else:  # 最后一次尝试失败
                        error_msg = f"API调用失败: {str(e)}"
                        print(error_msg)
                        yield f"data: {json.dumps({'content': f'**{error_msg}**'})}\n\n"
                        return
            
            # 如果所有尝试都失败且没有返回任何内容
            if not full_response.strip():
                error_msg = "API返回数据为空，请稍后重试"
                print(error_msg)
                yield f"data: {json.dumps({'content': f'**{error_msg}**'})}\n\n"
        
        return Response(generate(), mimetype='text/event-stream')
    
    except Exception as e:
        error_msg = f"发生错误: {str(e)}"
        print(error_msg)
        return jsonify({"error": error_msg}), 500

# Add this new endpoint to page.py
@deepseek_bp.route('/reset_chat', methods=['POST'])
def reset_chat():
    global messages
    messages = [{"role": "system", "content": "你是一个严谨的AI助手"}]
    return jsonify({"status": "success"})

@deepseek_bp.route('/er_upload', methods=['POST'])
def er_upload():
    task_id = request.form['task_id']
    print("111111111111111111111")
    print(task_id)
    data_str = None
    global prompt_new
    prompt_new =  request.form.get("prompt")
    if task_id == '':
        if 'upload_data' not in request.files:
            print('没有文件部分')
        file = request.files['upload_data']  
        if not file.filename.endswith('.txt'):
            return jsonify({'success': False, 'message': '请导入txt文件'})
        sep = request.form.get("sep", '')  
        data = []
        try:
            # 读取文件内容并解码为UTF-8字符串
            file_content = file.read().decode('utf-8')
            # 根据分隔符处理内容
            data = []
            if sep == '':  # 当sep为空时，按行分割
                lines = file_content.split('\n')  # 按换行符分割
                for line in lines:
                    line = line.strip()  # 去除首尾空白字符
                    if line:  # 跳过空行
                        data.append(line)
            else:  # 当sep不为空时，按指定分隔符分割
                # 先按行分割，再对每行按sep分割
                data = file_content.split(sep)
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'文件处理失败: {str(e)}'
            })
        data_str = '///'.join(data)
    else:
        try:
            datalist = TaskDataRepository.get_data_by_task(task_id)
            print(f"datalist:{datalist}")
            print(f"textlist:{[data.text for data in datalist]}")
            data_str = '///'.join([data.text for data in datalist])
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'舆情数据查询失败: {str(e)}'
            })
    global msg 
    if data_str!= '':
        msg = prompt_new + data_str
        return jsonify({'success': True, 'redirect': '/page/deepseek/home'})
    else:
        return jsonify({
            'success': False,
            'message': f'导入数据为空'
        }) 

@deepseek_bp.route('/get_preset_message')
def get_preset_message():
    # 这里可以从数据库或配置文件中读取预设信息
    return jsonify({"preset_message": sumup_prompt})

