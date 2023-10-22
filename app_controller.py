import base64
import io
import json
import re, PyPDF2
import tempfile
import typing
import time
from datetime import datetime
import pytesseract
import PIL
from flask import render_template, redirect, request, app, flash, send_file
from io import BytesIO

from werkzeug.utils import secure_filename
from tempfile import gettempdir

from app_model import appModel
from pdfminer.high_level import extract_pages, extract_text
from PIL.Image import Image
import smtplib
import os, pyttsx3, speech_recognition
import subprocess

class appController():
    def __init__(self, app):
        pass

    def register(self):
        if request.method == "POST":
            if request.form.get("usreg"):
                name = request.form.get("fname")
                usname = request.form.get("usname")
                usemail = request.form.get("usemail")
                usnum = request.form.get("usphnum")
                uspwd = request.form.get("uspwd")

                am = appModel(app)
                am.addUser(name, usname, usemail, usnum, uspwd)

                #self.send_email(usemail, name, usname)

                return redirect("/")

    def send_email(self, usemail, name, usname):
        EMAIL_ADDRESS = os.environ.get('MAIL_DEFAULT_SENDER')
        EMAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')

        with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()

            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)

            subject = 'Registeration Success'
            body = f"""Dear Customer \n {name}, Thanks For Registering with Hemankit. 
                Please use the following Username to log-in to the application:{usname} \n 
                Thank you for using our application. \n 
                For any further queries please contact +1000000000"""

            msg = f'Subject: {subject}\n\n {body}'

            smtp.sendmail(EMAIL_ADDRESS, usemail, msg)

    def showNewProject(self):
        return render_template("new-project.html")

    def userCode(self):
        if request.method == "GET":
            code = request.args.get("code")
            result = self.performance_analysis(code)
            language = request.args.get("language")
            print(code, language)
            command = []
            if language == "python":
                command = ["python3", "-m", "temp.py"]
            with open("temp.py", "w") as f:
                f.write(code)
            print(command)
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output = process.stdout.decode("utf-8")
            error = process.stderr.decode("utf-8")
            print(output, error)
            if not output:
                return json.dumps(error)
            else:
                return json.dumps(output)

    def uploader(self, file):

            filename = secure_filename(file.filename)

            filepath = os.path.join(tempfile.gettempdir(), filename)

            file.save(filepath)
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)

                page = reader.pages[0]

                text = page.extract_text()

            return render_template("new-project.html", text=text)

    def text_to_speech(self):
         if request.method == "GET":
             test = request.args.get("tts")
             print(test)
             text_speech = pyttsx3.init()
             text_speech.say(f"{test}")
             text_speech.runAndWait()
             return None

    def speech_recognition(self):
         if request.method == "GET":
             recognizer = speech_recognition.Recognizer()
             while True:
                 try:
                     with speech_recognition.Microphone() as mic:
                         recognizer.adjust_for_ambient_noise(mic, duration=0.5)
                         audio = recognizer.listen(mic)
                         text = recognizer.recognize_google(audio)
                         text = text.lower()
                         print(f"Recognized {text}")
                         return json.dumps(text)
                 except speech_recognition.UnknownValueError:
                    # Reset the recognizer
                    recognizer = speech_recognition.Recognizer()
                    # Continue listening
                    continue
                 # Reinitialize the text variable
                 text = ""

    def pdf_recognition(self,pdf: PyPDF2.PdfFileReader):
        text = extract_text(pdf)
        print(text)

    def image_recognition(self):
        if request.method == 'POST':
            file = request.files['file']
            filename = secure_filename(file.filename)

            filepath = os.path.join(tempfile.gettempdir(), filename)

            file.save(filepath)
            myconfig = r"--psm 6 --oem 3"

            with open(filepath, "rb") as f:

                text = pytesseract.image_to_string(PIL.Image.open(f), config=myconfig)
            print(text)
            return render_template("new-project.html", text=text)

    def performance_analysis(self, code):
        print()




