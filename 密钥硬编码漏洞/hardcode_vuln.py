# hardcode_vuln.py 
import requests

API_BASE = "https://api.example.com"
API_KEY  = "sk_live_1a2b3c4d5e6f"   # ⚠️ 硬编码！一旦提交到仓库就会被永久记录
# 正确做法
# import os
# API_KEY  = os.environ["PAY_API_KEY"]      # 运行时注入（如 K8s Secret / CI 注入）

def fetch_profile(uid: str):
    r = requests.get(
        f"{API_BASE}/v1/users/{uid}",
        headers={"Authorization": f"Bearer {API_KEY}"}
    )
    return r.json()

