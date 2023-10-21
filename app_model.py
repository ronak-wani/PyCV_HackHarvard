import sqlite3

class appModel():
    def __init__(self, app):
        self.con = sqlite3.connect("guiviz.db")
        self.cur = self.con.cursor()
        self.cur.execute("""CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, name TEXT NOT NULL,
        useremail TEXT NOT NULL, usernumber TEXT NOT NULL, password TEXT NOT NULL)""")

    def loginDetails(self, username):
        self.cur.execute("SELECT password FROM users WHERE username=?", (username,))
        pwdata = self.cur.fetchall()
        return pwdata

    def addUser(self, name, usname, usemail, usnum, uspwd):
        self.cur.execute("INSERT INTO users (username, name, useremail, usernumber, password) VALUES (?,?,?,?, ?)",(usname, name, usemail, usnum, uspwd))
        self.con.commit()

    def __del__(self):
        self.cur.close()
        self.con.close()