# 128-bit 不可预测 ID（base32url 编码）
import secrets, base64
raw = secrets.token_bytes(16)  # 128-bit
id_ = base64.urlsafe_b64encode(raw).rstrip(b'=').decode()
print(id_)  # e.g., "3uuqUuP3xbzB9g7e9Bf7XQ"
# ⚠️ 该 ID 仅不可预测，但不防伪（可被伪造）