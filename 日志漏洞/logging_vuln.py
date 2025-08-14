from flask import Flask, request
import logging, traceback

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route("/login", methods=["POST"])
def login():
    # print("CT:", request.headers.get("Content-Type"))
    # print("RAW:", request.get_data(as_text=True))
    data = request.get_json(silent=True) or {}
    # ❌ 把原始请求体直接打日志（包含密码/令牌）
    app.logger.info("login attempt: %s", data)

    # 模拟内部异常（比如 DB 超时）
    raise RuntimeError(f"DB timeout while querying users table with payload={data}")

# ❌ 错误处理：把内部异常和堆栈回传给客户端（信息泄露）
@app.errorhandler(Exception)
def on_error(e):
    app.logger.error("internal error: %s", traceback.format_exc())
    return f"Internal error: {e}\n{traceback.format_exc()}", 500

if __name__ == "__main__":
    app.run()  # http://127.0.0.1:5000

# 测试命令：
# Linux/Mac:
# curl -s -X POST http://127.0.0.1:5000/login \
#   -H 'Content-Type: application/json' \
#   -d '{"username":"alice","password":"SuperSecret123","otp":"123456"}'
# Windows:
# Invoke-RestMethod -Method POST -Uri http://127.0.0.1:5000/login `
#   -ContentType 'application/json' `
#   -Body '{"username":"alice","password":"SuperSecret123","otp":"123456"}'

