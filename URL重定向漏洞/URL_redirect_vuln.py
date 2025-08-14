# vuln_open_redirect.py  —— 漏洞演示：切勿在生产中这样写
from flask import Flask, request, redirect

app = Flask(__name__)

@app.route("/")
def index():
    return "URL重定向漏洞演示. 试一下： http://127.0.0.1:5000/login?next=https://www.baidu.com"

@app.route("/login")
def login():
    # ❶ 直接使用用户提供的 next 参数做重定向（危险！）
    next_url = request.args.get("next", "/")
    return redirect(next_url)  # ⚠️ 未校验 → Open Redirect

if __name__ == "__main__":
    app.run(debug=True)
