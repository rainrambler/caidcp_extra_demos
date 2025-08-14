import ast, textwrap, difflib

DANGEROUS_APIS = {("subprocess", "run"), ("os", "system")}

def uses_user_input(arg_src: str) -> bool:
    # 极简启发式：把包含 "input(" 或 "request.args" 视作可能受控输入
    return "input(" in arg_src or "request.args" in arg_src or "sys.argv" in arg_src

def find_dangerous_calls(src: str):
    """返回疑似危险调用列表（函数名、是否shell=True、参数源）"""
    out = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # 获取函数全名
            fn = node.func
            name = None
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
                name = (fn.value.id, fn.attr)  # e.g. ('subprocess','run')
            elif isinstance(fn, ast.Name):
                name = (None, fn.id)
            # 检查危险 API
            if name in DANGEROUS_APIS or name == (None, "system"):
                # 检查关键字参数 shell=True
                shell_true = any(isinstance(k, ast.keyword) and k.arg == "shell" and getattr(k.value, "value", None) is True
                                 for k in node.keywords)
                # 抽取第一个位置参数的源码片段（非常粗糙，仅演示）
                arg_src = ast.get_source_segment(src, node.args[0]) if node.args else ""
                out.append({"api": name, "shell_true": shell_true, "arg_src": arg_src})
    return out

def causal_intervention_shell_false(src: str) -> str:
    """把 shell=True 改成 False（do-操作），查看告警是否消失"""
    class Rewriter(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            # 改写 shell=True -> shell=False
            for kw in node.keywords or []:
                if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    kw.value = ast.Constant(False)
            return node
    tree = ast.parse(src)
    new_src = ast.unparse(Rewriter().visit(tree))  # Python 3.9+
    return new_src

def explain_with_causality(src: str):
    baseline = find_dangerous_calls(src)
    if not baseline:
        return {"vulnerable": False, "reason": "no dangerous api"}
    # 是否存在“用户输入 + shell=True”
    vuln = any(call["shell_true"] and uses_user_input(call["arg_src"]) for call in baseline)
    if not vuln:
        return {"vulnerable": False, "reason": "no user-controlled shell=True"}
    # 干预：强行把 shell=True 变成 False，再检测
    new_src = causal_intervention_shell_false(src)
    after = find_dangerous_calls(new_src)
    vuln_after = any(call["shell_true"] and uses_user_input(call["arg_src"]) for call in after)
    return {
        "vulnerable": True,
        "causal": not vuln_after,  # 干预后消失 => shell=True 是致因
        "fix_suggestion": "set shell=False and pass args list; or sanitize/whitelist input",
        "patched_preview": new_src
    }

def differential_report(ai_src: str, secure_ref: str):
    """差分：只关注高风险构造是否仅出现在 AI 代码里"""
    ai_calls = find_dangerous_calls(ai_src)
    ref_calls = find_dangerous_calls(secure_ref)
    risky_ai = [c for c in ai_calls if c["shell_true"] or uses_user_input(c["arg_src"])]
    risky_ref = [c for c in ref_calls if c["shell_true"] or uses_user_input(c["arg_src"])]
    diff = list(difflib.unified_diff(
        secure_ref.splitlines(), ai_src.splitlines(),
        fromfile="secure_ref.py", tofile="ai_code.py", lineterm=""
    ))
    return {"ai_risky_calls": risky_ai, "ref_risky_calls": risky_ref, "code_diff": "\n".join(diff)}

# === Demo ===
AI_CODE = textwrap.dedent("""
    import subprocess, sys
    cmd = sys.argv[1]                       # 模拟用户输入
    subprocess.run(cmd, shell=True)         # 可注入
""")

SECURE_REF = textwrap.dedent("""
    import subprocess, sys, shlex
    cmd = sys.argv[1]
    subprocess.run(["/usr/bin/ls", "--", cmd])    # 参数列表，不走 shell
""")

if __name__ == "__main__":
    print("==因果解释==")
    print(explain_with_causality(AI_CODE))
    print("==差分报告==")
    rep = differential_report(AI_CODE, SECURE_REF)
    print("AI 风险调用:", rep["ai_risky_calls"])
    print("参考实现风险调用:", rep["ref_risky_calls"])
    print(rep["code_diff"])
