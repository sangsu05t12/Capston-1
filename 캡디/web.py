import os
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from threading import Thread

def run_server():
    os.chdir("c:/Users/user/OneDrive - 동의대학교/바탕 화면/3학년/2/웹프/캡디")
    server_address = ("", 8000)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    httpd.serve_forever()

def open_browser():
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    server_thread = Thread(target=run_server)
    server_thread.start()

    open_browser()