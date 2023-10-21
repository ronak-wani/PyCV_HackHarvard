from flask import Flask, redirect, render_template, request, session, app
from app_model import appModel

user = ""

class Login():
    def __init__(self, app):
        pass

    def get_user(self):
        global user
        return user

    def showLoginPage(self):
        return render_template("user_login.html")

    def showRegister(self):
        return render_template("register.html")

    def index(self):
        global user
        if not session.get("name"):
            return render_template("/login.html")
        user = session.get("name")
        return render_template("index.html")

    def showLogin(self):
        return render_template("login.html")

    def login(self):

        global user
        errorpwd = ""
        incorrectpwd = ""
        if request.method == "POST":
            if request.form.get("login"):
                session["name"] = request.form.get("name")
                upwd = request.form.get("pwd")
                user = session["name"]
                am = appModel(app)
                pwdata = am.loginDetails(session["name"])
                if not pwdata:
                    errorpwd = "User does not exist. Please Register"
                    return render_template("login.html", error=errorpwd)
                elif upwd == pwdata[0][0]:
                    return render_template("index.html")
                else:
                    incorrectpwd = "Wrong Password. Please Try Again"
                    return render_template("login.html", error=incorrectpwd)

        return render_template("user_login.html")

    def logout(self):
        session["name"] = None
        return redirect("/show-loginfront")
