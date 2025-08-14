from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def index():
    q = request.args.get("q", "World")
    # ⚠️ 直接把用户输入拼进 HTML，未做任何转义 → XSS
    return f"<h1>Hello {q}</h1>"

if __name__ == "__main__":
    app.run()  # http://127.0.0.1:5000/

# 漏洞演示：请勿在生产环境中使用
# http://127.0.0.1:5000/?q=<script>alert('XSS')</script>
