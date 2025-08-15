import os
import tempfile
import secrets
from flask import Flask, request, redirect, url_for, flash, render_template
from werkzeug.utils import secure_filename
from flask_wtf.csrf import CSRFProtect
from flask_talisman import Talisman

# 定义上传目录和允许的文件扩展名
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg'}

# 定义允许的文件类型的Magic Number
# PNG: 89 50 4E 47 0D 0A 1A 0A
# JPG: FF D8 FF
MAGIC_NUMBERS = {
    '.png': b'\x89PNG\r\n\x1a\n',
    '.jpg': b'\xff\xd8\xff',
}

app = Flask(__name__)

# --- 安全配置 ---

# 1. 从环境变量获取密钥，如果不存在则生成随机密钥
app.secret_key = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(32))

# 2. 文件大小限制 (16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 3. CSRF保护
csrf = CSRFProtect(app)

# 4. 安全响应头
talisman = Talisman(app, 
    content_security_policy={
        'default-src': "'self'",
        'img-src': "'self' data:",
    },
    force_https=False # 在本地开发中禁用HTTPS强制跳转
)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传目录存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_safe_file_type(file_storage):
    """
    通过读取文件头（Magic Number）来验证文件类型，防止伪造扩展名。
    """
    try:
        header = file_storage.read(8)
        file_storage.seek(0)
        file_ext = '.' + file_storage.filename.rsplit('.', 1)[1].lower()
        
        if file_ext in MAGIC_NUMBERS:
            return header.startswith(MAGIC_NUMBERS[file_ext])
        return False
    except Exception as e:
        app.logger.error(f"Error reading file header: {e}")
        return False

@app.after_request
def add_security_headers(response):
    """添加额外的安全头"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # 检查文件大小 (通过MAX_CONTENT_LENGTH隐式处理，但可以添加显式检查)
        # Flask会在文件过大时抛出RequestEntityTooLarge异常，由框架处理
        # 我们也可以在这里捕获它或提前检查
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            
            if not is_safe_file_type(file):
                flash('Invalid file type. Only .png and .jpg are allowed.')
                return redirect(request.url)

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            if os.path.exists(filepath):
                flash(f'File {filename} already exists.')
                return redirect(request.url)

            try:
                file.save(filepath)
                flash(f'File "{filename}" uploaded successfully.')
                return redirect(url_for('upload_file'))
            except Exception as e:
                flash(f'Error saving file: {e}')
                return redirect(request.url)

        else:
            flash('File extension not allowed.')
            return redirect(request.url)
            
    return render_template('index.html')

if __name__ == '__main__':
    # 5. 默认禁用调试模式，通过环境变量控制
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode)
