# ws_server_vuln.py  —— 漏洞演示：请勿用于生产
import asyncio, subprocess
import websockets

async def handler(websocket):
    await websocket.send("Send me a path to list")
    user_input = await websocket.recv()
    cmd = f"ls {user_input}"  # ⚠️ 拼接到 shell 命令中
    print("Running:", cmd)
    # ⚠️ 根因：shell=True + 未经验证的用户输入
    subprocess.run(cmd, shell=True)
    await websocket.send("Done")

async def main():
    async with websockets.serve(handler, "127.0.0.1", 8765):
        print("Listening: ws://127.0.0.1:8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
