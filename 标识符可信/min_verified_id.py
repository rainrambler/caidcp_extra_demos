# 生成：payload = 版本(1B) + 时间戳(4B) + 随机(16B)
import time, secrets, hmac, hashlib, base64, struct
KEY = b'server-side-secret-32B-min'  # 存在 KMS/环境变量，不落库

def mint_id():
    payload = struct.pack(">BI", 1, int(time.time())) + secrets.token_bytes(16)
    tag = hmac.new(KEY, payload, hashlib.sha256).digest()[:16]  # 128-bit MAC
    tok = base64.urlsafe_b64encode(payload + tag).rstrip(b'=').decode()
    return tok

def verify_id(tok: str) -> bool:
    raw = base64.urlsafe_b64decode(tok + '==')
    payload, tag = raw[:-16], raw[-16:]
    exp = hmac.new(KEY, payload, hashlib.sha256).digest()[:16]
    return hmac.compare_digest(exp, tag)

tid = mint_id()
assert verify_id(tid)
print("New ID:", tid)