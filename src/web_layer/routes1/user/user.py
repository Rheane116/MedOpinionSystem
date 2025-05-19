from flask import Flask, session, render_template, redirect, Blueprint, request, jsonify
from src.service_layer.system_service import *

def get_password_hash(password: str):
    return hashlib.sha256(password.encode()).hexdigest()

user_bp = Blueprint('user', __name__, url_prefix='/user', template_folder='.')

@user_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    else:
        username = request.form.get("username")
        password = request.form.get("password")

        try:
            AuthService.login(username, password)
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})

        session['username'] = username
        user = AuthService.get_user_by_username(username)
        session['user_id'] = user.user_id
        session['user_role'] = user.role

        return jsonify({'success': True, 'redirect': '/page/home'})

@user_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    else:
        username = request.form.get("username")
        password = request.form.get("password")
        if password != request.form['check_password']:
            return jsonify({'success': False, 'message': '两次输入的密码不相符'})
        
        try:
            AuthService.register(username, password)
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
        
        return jsonify({'success': True, 'redirect': '/user/login'})



# 建议使用POST方法更安全（需修改前端）：
@user_bp.route('/logout', methods=['POST'])
def logout_post():
    session.clear()
    return redirect('/user/login')



