import os
import tempfile
import shutil
import logging
from secrets import token_hex
from flask import Flask, request, redirect, url_for, flash, render_template
from werkzeug.utils import secure_filename
from PIL import Image

# --- 配置 ---
# UPLOAD_FOLDER 是文件将要被存储的目录
# 请确保这个目录存在，并且Web服务器用户有写入权限。
# 为了安全，这个目录不应该在Web服务器的文档根目录下。
UPLOAD_FOLDER = 'uploads'
# 允许上传的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg'}
# 最大文件大小 (字节) 5MB
MAX_CONTENT_LENGTH = 5 * 1024 * 1024
MAX_FILENAME_LENGTH = 100
# 文件头部标识 (Magic Numbers)
MAGIC_NUMBERS = {
    '.png': b'\x89PNG\r\n\x1a\n',
    '.jpg': b'\xff\xd8\xff',
}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# 安全获取或生成会话密钥（避免硬编码）
app.secret_key = os.getenv('FLASK_SECRET_KEY', token_hex(32))

# 日志配置
logging.basicConfig(
    filename='upload.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

def log_upload_attempt(filename, success, error=None):
    if success:
        logging.info(f'文件上传成功: {filename}')
    else:
        logging.warning(f'文件上传失败: {filename}, 原因: {error}')

# --- 辅助函数 ---

def allowed_file(filename):
    """检查文件扩展名是否被允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_safe_path(path):
    """防止路径遍历攻击"""
    # os.path.abspath会规范化路径，例如 'a/b/../c' -> 'a/c'
    # 我们将检查规范化后的路径是否仍然在预期的上传目录内
    base_dir = os.path.abspath(app.config['UPLOAD_FOLDER'])
    target_path = os.path.abspath(os.path.join(base_dir, path))
    return os.path.commonpath([base_dir, target_path]) == base_dir

def get_file_type(file_stream):
    """通过读取文件头来识别真实的文件类型"""
    header = file_stream.read(8) # 读取前8个字节
    file_stream.seek(0) # 重置文件流指针
    for ext, magic in MAGIC_NUMBERS.items():
        if header.startswith(magic):
            return ext
    return None

def validate_image_file(file_path: str) -> bool:
    """使用Pillow验证图片文件完整性"""
    try:
        with Image.open(file_path) as img:
            img.verify()  # 验证文件
        return True
    except Exception:
        return False

def verify_upload_directory_permissions():
    """验证上传目录权限(基础检查)"""
    upload_path = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_path):
        return
    try:
        mode = oct(os.stat(upload_path).st_mode)[-3:]
        # 仅做示例，允许常见的755/750/750/700等
        if mode not in ['755', '750', '750', '700', '740', '744']:
            logging.warning(f'上传目录权限({mode}) 可能不符合最小权限原则，请人工复核')
    except Exception as e:
        logging.warning(f'无法验证上传目录权限: {e}')

# --- 路由 ---

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 1. 检查请求中是否包含文件部分
        if 'file' not in request.files:
            flash('请求中没有文件部分')
            return redirect(request.url)
        
        file = request.files['file']

        # 2. 如果用户没有选择文件，浏览器可能会提交一个没有文件名的空部分
        if file.filename == '':
            flash('未选择文件')
            return redirect(request.url)

        # 3. 文件名和类型验证
        if file and file.filename and allowed_file(file.filename):
            # 3.1 清理文件名，防止路径遍历等攻击
            filename = secure_filename(file.filename)

            # 文件名长度限制
            if len(filename) > MAX_FILENAME_LENGTH:
                flash('文件名过长')
                log_upload_attempt(filename, False, '文件名过长')
                return redirect(request.url)
            
            # 3.2 再次验证路径安全性
            if not is_safe_path(filename):
                flash('检测到不安全的路径')
                return redirect(request.url)

            # 4. 使用安全的临时文件进行操作
            temp_fd, temp_path = tempfile.mkstemp()
            try:
                file.save(temp_path)

                # 4.1 验证文件头 (Magic Number)
                with open(temp_path, 'rb') as f:
                    real_ext = get_file_type(f)

                if real_ext is None:
                    flash('无法识别文件类型')
                    log_upload_attempt(filename, False, '无法识别类型')
                    return redirect(request.url)

                expected_ext = '.' + filename.rsplit('.', 1)[1].lower()
                if real_ext != expected_ext:
                    flash('文件扩展名与内容不匹配')
                    log_upload_attempt(filename, False, f'扩展名{expected_ext}≠内容{real_ext}')
                    return redirect(request.url)

                # Pillow验证
                if not validate_image_file(temp_path):
                    flash('无效的图片文件')
                    log_upload_attempt(filename, False, 'Pillow验证失败')
                    return redirect(request.url)

                # 5. 验证通过，将文件从临时位置移动到最终位置
                final_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                shutil.move(temp_path, final_path)
                
                # 6. 权限控制 (提醒)
                # 确保上传目录不可执行。这通常在Web服务器或操作系统级别配置。
                # 例如，在Linux上，可以运行 `chmod -R -x+w uploads`
                # 这里我们只打印一条消息作为提醒
                print(f"提醒: 请确保 '{app.config['UPLOAD_FOLDER']}' 目录没有执行权限。")

                flash(f'文件 "{filename}" 上传成功')
                log_upload_attempt(filename, True)
                return redirect(url_for('upload_file'))

            except Exception as e:
                flash('上传过程中发生错误')
                log_upload_attempt(file.filename or '未知文件', False, str(e))
                return redirect(request.url)
            finally:
                # 4.2 确保临时文件被清理
                if os.path.exists(temp_path):
                    os.close(temp_fd)
                    os.remove(temp_path)
        else:
            flash('不允许的文件类型')
            try:
                original_name = request.files['file'].filename if 'file' in request.files and request.files['file'].filename else '未知文件'
            except Exception:
                original_name = '未知文件'
            log_upload_attempt(original_name, False, '类型不被允许')
            return redirect(request.url)

    return render_template('index.html')

if __name__ == '__main__':
    # 确保上传目录存在
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    verify_upload_directory_permissions()
    # 生产部署请使用 WSGI/ASGI 服务器 (gunicorn/uwsgi) 并通过环境变量控制 debug
    app.run(debug=bool(os.getenv('FLASK_DEBUG', '0') == '1'))

@app.errorhandler(Exception)
def handle_error(error):
    logging.error(f'未处理异常: {error}')
    return '服务器内部错误，请稍后重试', 500
