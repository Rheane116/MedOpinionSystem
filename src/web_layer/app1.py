from flask import Flask, session, request, redirect, render_template
import re
from src.data_access_layer.database import db, init_db
from config.config import Config

from .routes1.page import page
from .routes1.user import user

app = Flask(__name__)
app.config.from_object(Config)
init_db(app)
app.register_blueprint(page.page_bp)
app.register_blueprint(user.user_bp)

@app.route('/')
def hello_world():  # put application's code here
    session.clear()
    return redirect('/user/login')

@app.before_request
def before_request():
    pattern = re.compile(r'^/static')
    if re.search(pattern, request.path):
        return
    elif request.path == '/user/login' or request.path == '/user/register':
        return
    elif session.get('username'):
        return
    return redirect('/user/login')

@app.route('/<path:path>')
def catch_all(path):
    return render_template('404.html')


if __name__ == '__main__':
    app.run(port=8187, debug=True)
    #app.run()
