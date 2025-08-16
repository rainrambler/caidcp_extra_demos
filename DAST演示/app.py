from flask import Flask, request, render_template

app = Flask(__name__)

# 一个简单的过滤器，移除<script>和</script>标签
def simple_xss_filter(input_string):
    """一个简单的XSS过滤器，它会移除<script>标签。"""
    if not input_string:
        return ""
    # 为了演示，这里只做一个非常简单的、不区分大小写的替换
    filtered_string = input_string.replace('<script>', '[removed]').replace('</script>', '[removed]')
    return filtered_string

@app.route('/', methods=['GET', 'POST'])
def index():
    user_input = ""
    if request.method == 'POST':
        # 从表单获取输入
        raw_input = request.form.get('user_input', '')
        # 应用过滤器
        user_input = simple_xss_filter(raw_input)
    
    # 未经充分转义就渲染用户输入，导致XSS漏洞
    # 在模板中，我们将使用 `| safe` 来确保HTML被渲染
    return render_template('index.html', user_input=user_input)

if __name__ == '__main__':
    # 监听所有网络接口，方便从外部访问
    app.run(host='0.0.0.0', port=5001, debug=True)
