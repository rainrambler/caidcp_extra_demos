# URL重定向漏洞验证报告

## 漏洞描述
该应用存在URL重定向漏洞，位于`URL_redirect_vuln.py`文件中。漏洞代码直接使用用户提供的`next`参数进行重定向，未经任何验证，可能导致开放重定向攻击。

## 验证步骤

1. 启动Flask应用：
```bash
python URL_redirect_vuln.py
```

2. 构造验证POC URL：
```
http://127.0.0.1:5000/login?next=https://www.baidu.com
```

3. 漏洞原理：
```python
@app.route("/login")
def login():
    next_url = request.args.get("next", "/")
    return redirect(next_url)  # 未经验证直接重定向
```

## 验证结果

1. 期望行为：应用应该只允许重定向到同一域名下的URL
2. 实际行为：应用允许重定向到任意外部域名
3. 影响：攻击者可以利用此漏洞将用户重定向到钓鱼网站

## 漏洞利用场景

1. 钓鱼攻击：攻击者可以构造一个指向恶意网站的链接
2. 社会工程：通过合法域名的重定向增加攻击的可信度

## POC演示命令
```bash
# 在浏览器中访问以下URL即可验证漏洞：
http://127.0.0.1:5000/login?next=https://www.baidu.com

# 或使用curl命令验证（注意观察响应头中的Location字段）：
curl -v http://127.0.0.1:5000/login?next=https://www.baidu.com
```

## 漏洞级别
高危（High）：
- 可被远程利用
- 可导致用户被重定向到恶意网站
- 可用于钓鱼攻击

## 修复建议

1. 实施URL白名单机制
2. 验证重定向URL的域名
3. 使用相对路径而非绝对URL
4. 采用url_for()函数生成内部URL

示例修复代码：
```python
from urllib.parse import urlparse
from flask import url_for

@app.route("/login")
def login():
    next_url = request.args.get("next", "/")
    # 验证URL
    parsed = urlparse(next_url)
    if parsed.netloc:  # 如果包含域名
        if parsed.netloc != request.host:  # 如果不是当前域名
            return "重定向URL不被允许", 400
    return redirect(next_url)
