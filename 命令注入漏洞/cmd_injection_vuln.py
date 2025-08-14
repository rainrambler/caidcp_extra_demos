# cmd_injection_vuln.py  —— 演示用，千万不要在生产中这样写
import subprocess, platform

def list_path(user_input: str):
    if platform.system() == "Windows":
        cmd = f"dir {user_input}"
    else:
        cmd = f"ls {user_input}"
    print("Running:", cmd)
    # ⚠️ 关键问题：shell=True + 未经过滤的拼接输入
    subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    # 正常输入（安全场景）
    # list_path(".")

    # 恶意输入（触发注入；仅在演示目录里试验）
    # Linux/macOS:
    #   .; touch injected.txt
    # Windows:
    #   list_path(". & echo INJECTED> injected.txt")
    list_path(". & echo 注入成功> injected.txt")