#!/usr/bin/env python3
"""Small browser visualizer for DDS odometry path."""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "third_party" / "unitree_sdk2_python"))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_


INDEX_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>DDS Odom Viz</title>
  <style>
    html, body { margin: 0; height: 100%; background: #111; color: #eee; font: 14px system-ui, sans-serif; }
    #bar { position: fixed; left: 0; top: 0; right: 0; padding: 8px 12px; background: #191919; border-bottom: 1px solid #333; display: flex; gap: 18px; align-items: center; }
    #status { font-weight: 700; }
    #status.stale { color: #ffb347; }
    #status.fresh { color: #7ee787; }
    canvas { display: block; width: 100vw; height: 100vh; }
    button { background: #2b2b2b; color: #eee; border: 1px solid #555; padding: 4px 8px; border-radius: 4px; cursor: pointer; }
  </style>
</head>
<body>
  <div id="bar">
    <span id="status">connecting</span>
    <span id="pose"></span>
    <span id="err"></span>
    <span id="count"></span>
    <button onclick="resetView()">Reset View</button>
  </div>
  <canvas id="c"></canvas>
<script>
const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const poseEl = document.getElementById("pose");
const errEl = document.getElementById("err");
const countEl = document.getElementById("count");
let path = [];
let start = null;
let latest = null;
let scale = 180;
let origin = {x: 0, y: 0};
let auto = true;

function resize() {
  canvas.width = window.innerWidth * devicePixelRatio;
  canvas.height = window.innerHeight * devicePixelRatio;
  canvas.style.width = window.innerWidth + "px";
  canvas.style.height = window.innerHeight + "px";
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  draw();
}
window.addEventListener("resize", resize);

function resetView() {
  auto = true;
  draw();
}

function wrapPi(a) {
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
}

function fitView() {
  if (!path.length) return;
  let minx = path[0].x, maxx = path[0].x, miny = path[0].y, maxy = path[0].y;
  for (const p of path) {
    minx = Math.min(minx, p.x); maxx = Math.max(maxx, p.x);
    miny = Math.min(miny, p.y); maxy = Math.max(maxy, p.y);
  }
  const w = Math.max(maxx - minx, 0.5);
  const h = Math.max(maxy - miny, 0.5);
  scale = Math.min((window.innerWidth * 0.72) / w, (window.innerHeight * 0.72) / h, 260);
  origin.x = (minx + maxx) / 2;
  origin.y = (miny + maxy) / 2;
}

function sx(x) { return window.innerWidth / 2 + (x - origin.x) * scale; }
function sy(y) { return window.innerHeight / 2 - (y - origin.y) * scale; }

function drawGrid() {
  ctx.strokeStyle = "#262626";
  ctx.lineWidth = 1;
  const step = 0.25;
  const xmin = origin.x - window.innerWidth / 2 / scale;
  const xmax = origin.x + window.innerWidth / 2 / scale;
  const ymin = origin.y - window.innerHeight / 2 / scale;
  const ymax = origin.y + window.innerHeight / 2 / scale;
  for (let x = Math.floor(xmin / step) * step; x <= xmax; x += step) {
    ctx.beginPath(); ctx.moveTo(sx(x), 0); ctx.lineTo(sx(x), window.innerHeight); ctx.stroke();
  }
  for (let y = Math.floor(ymin / step) * step; y <= ymax; y += step) {
    ctx.beginPath(); ctx.moveTo(0, sy(y)); ctx.lineTo(window.innerWidth, sy(y)); ctx.stroke();
  }
  ctx.strokeStyle = "#555";
  ctx.beginPath(); ctx.moveTo(sx(0), 0); ctx.lineTo(sx(0), window.innerHeight); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, sy(0)); ctx.lineTo(window.innerWidth, sy(0)); ctx.stroke();
}

function drawRobot(p) {
  const x = sx(p.x), y = sy(p.y);
  const yaw = p.yaw || 0;
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(-yaw);
  ctx.fillStyle = "#7ee787";
  ctx.beginPath();
  ctx.moveTo(14, 0); ctx.lineTo(-9, -7); ctx.lineTo(-9, 7); ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function draw() {
  if (auto) fitView();
  ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
  drawGrid();
  if (start) {
    ctx.fillStyle = "#58a6ff";
    ctx.beginPath(); ctx.arc(sx(start.x), sy(start.y), 6, 0, Math.PI * 2); ctx.fill();
  }
  if (path.length > 1) {
    ctx.strokeStyle = "#ff7b72";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(sx(path[0].x), sy(path[0].y));
    for (const p of path) ctx.lineTo(sx(p.x), sy(p.y));
    ctx.stroke();
  }
  if (latest) drawRobot(latest);
}

const es = new EventSource("/events");
es.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.type === "reset") {
    path = []; start = null; latest = null; draw(); return;
  }
  latest = msg;
  if (!start) start = msg;
  path.push(msg);
  if (path.length > 5000) path.shift();
  const age = msg.age;
  statusEl.textContent = age <= msg.stale ? "fresh" : "STALE";
  statusEl.className = age <= msg.stale ? "fresh" : "stale";
  poseEl.textContent = `x=${msg.x.toFixed(3)} y=${msg.y.toFixed(3)} yaw=${msg.yaw.toFixed(3)} age=${age.toFixed(2)}s`;
  const dx = msg.x - start.x, dy = msg.y - start.y, dyaw = wrapPi(msg.yaw - start.yaw);
  errEl.textContent = `from start: dx=${dx.toFixed(3)} dy=${dy.toFixed(3)} dist=${Math.hypot(dx, dy).toFixed(3)} dyaw=${dyaw.toFixed(3)}`;
  countEl.textContent = `n=${msg.count}`;
  draw();
};
es.onerror = () => { statusEl.textContent = "disconnected"; statusEl.className = "stale"; };
resize();
</script>
</body>
</html>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("net", nargs="?", default="eno1")
    p.add_argument("--topic", default="rt/dodge/odom")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--stale", type=float, default=0.35)
    args = p.parse_args()

    ChannelFactoryInitialize(0, args.net)
    state = {"data": None, "recv": 0.0, "count": 0}
    clients: list[queue.Queue] = []

    def cb(msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        state["data"] = data
        state["recv"] = time.time()
        state["count"] += 1
        payload = {
            "x": float(data.get("x", 0.0)),
            "y": float(data.get("y", 0.0)),
            "yaw": float(data.get("yaw", 0.0)),
            "age": 0.0,
            "stale": args.stale,
            "count": state["count"],
            "source": data.get("source", "?"),
        }
        for q in clients[:]:
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass

    sub = ChannelSubscriber(args.topic, String_)
    sub.Init(cb, 10)

    def age_updater():
        while True:
            time.sleep(0.5)
            data = state["data"]
            if data is None:
                continue
            payload = {
                "x": float(data.get("x", 0.0)),
                "y": float(data.get("y", 0.0)),
                "yaw": float(data.get("yaw", 0.0)),
                "age": time.time() - state["recv"],
                "stale": args.stale,
                "count": state["count"],
                "source": data.get("source", "?"),
            }
            for q in clients[:]:
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    pass

    threading.Thread(target=age_updater, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            return

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = INDEX_HTML.encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/events":
                q: queue.Queue = queue.Queue(maxsize=100)
                clients.append(q)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                try:
                    while True:
                        msg = q.get(timeout=15.0)
                        line = "data: " + json.dumps(msg, separators=(",", ":")) + "\n\n"
                        self.wfile.write(line.encode())
                        self.wfile.flush()
                except Exception:
                    pass
                finally:
                    if q in clients:
                        clients.remove(q)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[viz] open http://{args.host}:{args.port}  topic={args.topic} net={args.net}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
