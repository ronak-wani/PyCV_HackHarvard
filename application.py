from flask import Flask, redirect, render_template, request, session
from flask_session import Session
import login
from app_model import appModel
import app_controller

# Initial command
app = Flask(__name__)

# For database connection
db = appModel("guiviz.db")

lp = login.Login(app)
ac = app_controller.appController(app)

app.add_url_rule("/", view_func=lp.index)
app.add_url_rule("/show-loginfront", view_func=lp.showLogin)
app.add_url_rule("/login", methods=['GET', 'POST'], view_func=lp.login)
app.add_url_rule("/show-login", view_func=lp.showLoginPage)
app.add_url_rule("/show-reg", view_func=lp.showRegister)
app.add_url_rule("/logout", view_func=lp.logout)
app.add_url_rule("/register", methods=['GET', 'POST'], view_func=ac.register)
app.add_url_rule("/new-project", view_func=ac.showNewProject)
app.add_url_rule("/user-code", methods=['GET', 'POST'], view_func=ac.userCode)
app.add_url_rule("/text-to-speech", methods=['GET', 'POST'], view_func=ac.text_to_speech)
app.add_url_rule("/speech-to-text", methods=['GET', 'POST'], view_func=ac.speech_recognition)
app.add_url_rule("/uploader", methods=['GET', 'POST'], view_func=ac.uploader)
app.add_url_rule("/uploader-img", methods=['GET', 'POST'], view_func=ac.image_recognition)
app.add_url_rule("/uploader", methods=['GET', 'POST'], view_func=ac.detect_file_type)

# Settings for creating session
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Command to run the application
if __name__ == "__main__":
    app.run()