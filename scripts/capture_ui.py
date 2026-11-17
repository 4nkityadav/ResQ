"""
scripts.capture_ui
==================
Captures high-resolution screenshots of the live Streamlit dashboard at
specified viewport dimensions using headless Chrome CDP.
"""

import sys
import time
import subprocess
import json
import base64
from pathlib import Path
import urllib.request
import urllib.error

try:
    import websocket
except ImportError:
    websocket = None


def capture(url: str, output_path: str, width: int = 1440, height: int = 900):
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    user_data = Path(r"C:\Users\pmax8\AppData\Local\Temp\chrome_snap_profile")
    user_data.mkdir(parents=True, exist_ok=True)
    
    port = 9223
    proc = subprocess.Popen([
        chrome_path,
        "--headless",
        "--disable-gpu",
        f"--remote-debugging-port={port}",
        f"--user-data-dir={str(user_data)}",
        "--no-first-run",
        "--no-default-browser-check"
    ])
    
    time.sleep(2.0)
    
    try:
        # Get websocket debugger URL
        tabs_url = f"http://127.0.0.1:{port}/json"
        req = urllib.request.urlopen(tabs_url, timeout=5)
        tabs = json.loads(req.read().decode())
        ws_url = tabs[0]["webSocketDebuggerUrl"]
        
        ws = websocket.create_connection(ws_url, timeout=10)
        
        # 1. Enable Page
        ws.send(json.dumps({"id": 1, "method": "Page.enable"}))
        ws.recv()
        
        # 2. Set viewport
        ws.send(json.dumps({
            "id": 2,
            "method": "Emulation.setDeviceMetricsOverride",
            "params": {
                "width": width,
                "height": height,
                "deviceScaleFactor": 1,
                "mobile": False
            }
        }))
        ws.recv()
        
        # 3. Navigate
        ws.send(json.dumps({"id": 3, "method": "Page.navigate", "params": {"url": url}}))
        ws.recv()
        
        # Wait for Streamlit rendering and plotly chart to mount
        start_wait = time.time()
        while time.time() - start_wait < 18.0:
            time.sleep(1.0)
            ws.send(json.dumps({
                "id": 99,
                "method": "Runtime.evaluate",
                "params": {"expression": "Boolean(document.querySelector('.js-plotly-plot') || document.querySelector('[data-testid=\"stPlotlyChart\"]'))", "returnByValue": True}
            }))
            res = json.loads(ws.recv())
            if res.get("id") == 99 and res.get("result", {}).get("result", {}).get("value") is True:
                time.sleep(1.0)  # Brief buffer for animations to settle
                break
        
        # 4. Capture screenshot
        ws.send(json.dumps({"id": 4, "method": "Page.captureScreenshot", "params": {"format": "png"}}))
        while True:
            res = json.loads(ws.recv())
            if res.get("id") == 4:
                img_b64 = res["result"]["data"]
                break
        
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "wb") as f:
            f.write(base64.b64decode(img_b64))
            
        print(f"Captured {width}x{height} to {output_path}")
        ws.close()
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


if __name__ == "__main__":
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 1440
    h = int(sys.argv[2]) if len(sys.argv) > 2 else 900
    out = sys.argv[3] if len(sys.argv) > 3 else f"docs/screenshots/phase2_ui_{w}.png"
    target_url = sys.argv[4] if len(sys.argv) > 4 else "http://localhost:8501"
    capture(target_url, out, w, h)
