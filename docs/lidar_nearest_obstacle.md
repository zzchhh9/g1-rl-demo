# 实时测最近障碍物距离

只用 LiDAR、不动机器人，让笔记本知道"前方/周围最近的障碍物有多远"。这一篇专门讲怎么写 `nearest_obstacle.py`——从原始点云到一个数字 + 一个方位角。

读这篇之前推荐先看 [Livox Mid-360 是怎么工作的](lidar_how_it_works.md)，至少知道：
- LiDAR 发的是 UDP，不是 ROS topic
- 每个 UDP 包是 36 字节头 + N 个 14 字节的点（`int32 xyz_mm + uint8 refl + uint8 tag`）
- 我们已经有一根从 Mid-360 → 笔记本的 UDP "水管"

---

## 整体设计

把任务拆三层：

```
┌────────────────────────────┐
│  3) main loop              │  每 100 ms 取最新点云、过滤、找最近、打印
│     (10 Hz)                │
├────────────────────────────┤
│  2) sensor → ego filter    │  z 范围 / r 范围 / 排除自身
├────────────────────────────┤
│  1) LivoxLidarReceiver     │  后台线程持续 recvfrom + 解析 + 写入 ring buffer
│     (always-on)            │  
└────────────────────────────┘
```

接收线程"一直跑"，主循环按需取**最新一段时间**的点云做处理。这是最简单的"生产者-消费者"模式：

- **生产者** = 接收线程，UDP 频率 ~2 kHz
- **消费者** = 主循环，按 10 Hz 处理
- **缓冲区** = `collections.deque(maxlen=N)` —— 自带满了丢老的，无锁竞争安全（Python GIL 保证 deque 写入原子）

---

## 1) 接收 + 解析

完整代码在 `test_lidar.py` 的 `LivoxLidarReceiver`。核心 30 行：

```python
class LivoxLidarReceiver:
    def __init__(self, port=56301, buffer_pts=50000):
        self._buf = collections.deque(maxlen=buffer_pts)
        self._lock = threading.Lock()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        self._sock.bind(("", port))
        self._sock.settimeout(0.5)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            data, _ = self._sock.recvfrom(65535)
            if len(data) < 36: continue
            ver, length, _ti, dot_num, _uc, _fc, dtype, _tt = \
                struct.unpack_from("<BHHHHBBB", data, 0)
            if ver != 0 or dtype != 1:  # 1 = CartesianHigh (mm)
                continue
            payload = data[36:]
            new = [struct.unpack_from("<iii", payload, i * 14)
                   for i in range(min(dot_num, len(payload) // 14))]
            with self._lock:
                for x, y, z in new:
                    self._buf.append((x * 0.001, y * 0.001, z * 0.001))

    @property
    def points(self):
        with self._lock:
            return np.array(self._buf, dtype=np.float32) if self._buf else None
```

几个工程细节值得说：

**SO_RCVBUF = 4 MB**：默认 Linux UDP 接收缓冲 ~200 KB，Mid-360 全速 ~280 KB/s 也勉强够，但如果主循环来不及处理偶尔会丢包。设到 4 MB 提供约 14 秒的吃水量，主循环就算卡住一两秒也不丢包。

**`SO_REUSEADDR`**：允许多个进程同时绑 `:56301`，调试时方便同时跑两个监听。

**`settimeout(0.5)`**：周期性醒过来检查停止信号，避免线程永远卡在 `recvfrom`。

**为什么用 deque 而不是 list**：`deque(maxlen=N)` 满了自动丢最旧的，O(1) 写。如果用 list 然后 `[-N:]`，每帧要拷贝 N 元素 = O(N) 浪费。

**`buffer_pts=50000` 怎么选**：Mid-360 点率 ~200k pts/s。50k 点 ≈ 0.25 秒数据。**这就是我们的"时间窗口"**——主循环看到的点云永远是最近 250 ms 累积的。窗口大了 → 点云密但延迟高；窗口小了 → 反应快但稀疏。

---

## 2) 过滤："哪些点算障碍物"

`pts` 拿到手是 N×3 的 numpy 数组（meters, sensor frame: **x = 前方、+y = 右侧、z = 上**；y 轴跟典型 ROS REP 105 相反——具体原因见 [lidar_how_it_works.md §6](lidar_how_it_works.md)）。所有过滤都是矢量化布尔索引：

```python
x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
r = np.sqrt(x*x + y*y)              # 水平距离

mask  = z > MIN_H                   # 排地面
mask &= z < MAX_H                   # 排天花板/高挂物
mask &= r > MIN_R                   # 排自身（机器人/支架）
mask &= r < MAX_R                   # 排远处噪点

valid = pts[mask]
```

四个阈值的物理含义：

| 阈值 | 默认 | 为什么 |
|---|---|---|
| `MIN_H` | **0.30 m** | LiDAR 在 G1 头顶 ~80 cm 高、向下倾斜；地面点的 `z`（sensor frame，向上为正）会接近 -0.8。设 0.30 把地面 + 矮台阶都过滤掉，只留"能撞到躯干"的高度 |
| `MAX_H` | **2.00 m** | 排天花板光源/吊灯，避免误检 |
| `MIN_R` | **0.30 m** | LiDAR 视野里能看到机器人自己的肩/臂/电源外壳——0.3m 内的点全是"自身"，丢 |
| `MAX_R` | **8.00 m** | 远处反射率低，点稀疏不可靠。**对避障而言 5–8m 足够**，超出意义不大 |

**关键观察**：所有过滤都在 **sensor frame**（不是世界坐标系）。原因——我们要做的是"我视野里最近有什么"，不需要全局定位。这让脚本独立于 IMU / 里程计，只要 LiDAR 来电就能用。

如果以后要做"绕开障碍走"那种用法（需要全局坐标），把 `pts` 用机器人姿态旋转一下即可（`deploy_dodge_real.py` 里的 `LidarObstacleDetector.detect()` 就是这么做的）：

```python
cy, sy = np.cos(robot_yaw), np.sin(robot_yaw)
wx = pts[:, 0] * cy - pts[:, 1] * sy + robot_pos[0]
wy = pts[:, 0] * sy + pts[:, 1] * cy + robot_pos[1]
wz = pts[:, 2] + robot_pos[2]
```

---

## 3) 找最近的点

过滤完拿到 `valid`，求最近就一行：

```python
rv = np.sqrt(valid[:, 0]**2 + valid[:, 1]**2)
idx = int(np.argmin(rv))
nearest = valid[idx]      # [x, y, z]
dist = float(rv[idx])
bearing = math.degrees(math.atan2(nearest[1], nearest[0]))
```

注意我们用**水平距离** `rv`（XY 平面）而不是 3D 距离 `np.linalg.norm`。原因：

- 障碍物对 G1 是否危险**取决于水平距离**——一个 1.8 m 高的人在你前面 1m，3D 距离可能 2m，但你撞不上"高度"，撞的是"宽度"
- 顶部/底部点对避障决策不影响，水平投影才是关键

如果需要 "最高/最低点"、"最具威胁的点" 等更复杂的判断，看 `nearest_obstacle.py` 里的注释扩展。

---

## 4) 完整主循环

```python
from test_lidar import LivoxLidarReceiver
import numpy as np, math, time

lidar = LivoxLidarReceiver(port=56301, buffer_pts=30000)  # 0.15s 窗口
time.sleep(0.3)

while True:
    pts = lidar.points
    if pts is None or len(pts) == 0:
        print("无数据"); time.sleep(0.1); continue

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.sqrt(x*x + y*y)
    mask = (z > 0.3) & (z < 2.0) & (r > 0.3) & (r < 8.0)
    valid = pts[mask]
    if len(valid) == 0:
        print("视野内无障碍"); time.sleep(0.1); continue

    rv = np.sqrt(valid[:, 0]**2 + valid[:, 1]**2)
    i = int(np.argmin(rv))
    p, d = valid[i], float(rv[i])
    b = math.degrees(math.atan2(p[1], p[0]))
    print(f"最近: {d:.2f}m  方位 {b:+.1f}°  xyz=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f})")
    time.sleep(0.1)
```

完整版（带 ANSI 着色、Ctrl+C 优雅退出、可选 matplotlib 可视化）见 `nearest_obstacle.py`。

---

## 5) 从零开始的完整流程

### 一次性准备（新机器/新环境只做一次）

```bash
# 装 sshpass（start_lidar.sh 用它免交互 SSH 到机器人）
sudo apt install -y sshpass

# 仓库已 clone 且 uv sync 完成的话，依赖都齐了
# 主要用到 numpy（uv 项目自带）和可选的 matplotlib
```

### 每次启动前的确认（10 秒）

```bash
# 网线插好，机器人开机后能 ping 通：
ping -c 2 192.168.123.164    # Jetson PC1（运 driver 和 forwarder 的地方）
ping -c 2 192.168.123.120    # Mid-360 LiDAR 本体

# 都成功才继续
```

### 启动数据流

```bash
cd ~/g1-rl-demo
./scripts/start_lidar.sh
```

成功输出长这样：

```
[1/4] preparing /tmp/livox-run on robot ...
[2/4] uploading MID360 config + forwarder + start scripts ...
[3/4] running driver briefly to tell LiDAR to start streaming ...
    ✓ LiDAR streaming started
[4/4] starting UDP forwarder ...
10326 python3 /tmp/livox-run/forward.py
✓ LiDAR pipeline up. Test from laptop:
    uv run python test_lidar.py
    uv run python test_lidar.py --duration 30 --detect
```

**说明**：这条命令在机器人侧只占两样东西，都在 `/tmp/livox-run/`（机器人重启清零）。
只要机器人不重启，LiDAR 会持续往笔记本推流，不需要重复跑。

### 跑工具

```bash
# 默认配置（推荐起手）
uv run python nearest_obstacle.py
```

默认参数：监听 `:56301`，高度过滤 `0.3–2.0m`，半径过滤 `0.3–8.0m`，
累积窗口 `0.10s`，刷新 `10 Hz`，颜色：红 < 1m / 黄 < 2m / 绿 > 2m。

输出长这样：

```
[ready] 监听点云数据… Ctrl+C 退出

t=  3.2s  pkts= 6088  buf=30000  最近: 0.63m  方位= -76.3°  (x=+0.15, y=-0.62, z=+0.32)  n= 312
```

按 `Ctrl+C` 退出。

### 常用调参

```bash
# 只看 5m 内，更短窗口（反应更快但点更稀疏）
uv run python nearest_obstacle.py --max-r 5.0 --window 0.05

# 更高的最小高度（机器人腿部容易在 0.3m 以下，避免误判）
uv run python nearest_obstacle.py --min-h 0.4

# 慢一点但更稳定（适合做演示）
uv run python nearest_obstacle.py --window 0.3 --rate 5

# 实时俯视图（额外开一个 matplotlib 窗口，需要图形界面）
uv run python nearest_obstacle.py --plot

# 去掉 ANSI 颜色（写入日志文件时用）
uv run python nearest_obstacle.py --no-color > log.txt
```

### 关闭

```bash
./scripts/stop_lidar.sh
```

### 怎么判断"现在能不能直接跑"

如果你不确定上一次的转发器是否还活着（比如刚回到工位、不知道之前会话什么状态），
先做一个 2 秒的 UDP 探测：

```bash
timeout 2 python3 -c "import socket; s=socket.socket(2,2); s.bind(('',56301)); s.settimeout(1.5); n=0
try:
  while True: s.recvfrom(2048); n+=1
except: pass
print('packets/1.5s =', n)"
```

- 输出 `packets/1.5s = 几千` → 数据流还在，直接 `uv run python nearest_obstacle.py`
- 输出 `packets/1.5s = 0` → 流断了，先 `./scripts/start_lidar.sh` 再跑工具

### 机器人或笔记本断电/重启后

转发器进程在重启后没了（机器人 `/tmp/livox-run/` 被清空，笔记本无所谓），
按下面顺序恢复：

```bash
# 1. 装 sshpass（只在新设备首次需要，永久有效）
sudo apt install -y sshpass

# 2. 确认网络（机器人开机就绪后）
ping -c 2 192.168.123.164

# 3. 启动 LiDAR 流水线
./scripts/start_lidar.sh

# 4. 跑工具
uv run python nearest_obstacle.py
```

### 排查

| 现象 | 检查 |
|---|---|
| `start_lidar.sh: command not found` | 没在 `g1-rl-demo` 仓库根目录运行 |
| `sshpass: not installed` | `sudo apt install -y sshpass` |
| `start_lidar.sh` 卡在 [3/4] 并报 `driver init failed` | 机器人侧 `bad_alloc`，重启机器人后重试；或 SSH 上去看 `/tmp/livox-run/driver.log` |
| `nearest_obstacle.py` 一直 "无数据" | 转发器没起：`sshpass -p 123 ssh unitree@192.168.123.164 'pgrep -af forward.py'`；或防火墙拦了 UDP :56301 |
| 距离读数跳变剧烈 | 累积窗口太小 → `--window 0.2` |
| 距离总是很小 (< 0.5m) 且方位固定 | LiDAR 看到机器人自身 → 提高 `--min-r` |

---

## 6) 性能 & 延迟

实测数据（笔记本 + G1 真机）：

| 指标 | 数值 |
|---|---|
| 输入包率 | **~2090 pkt/s** |
| 输入点率 | **~200k pts/s** |
| 主循环刷新率 | 10 Hz（可调到 50+ Hz） |
| 端到端延迟（LiDAR 看到 → Python `points` 反映） | **~50–100 ms** |
| CPU 占用（笔记本上） | < 5% （单核） |

延迟主要由"累积窗口 (`buffer_pts/point_rate`)"决定。如果要求 < 50ms 反应，把 `buffer_pts` 降到 5000-10000；但点云会变稀疏（< 1000 点/帧），需要相应放松障碍物检测的"点数阈值"。

---

## 7) 已知的坑

1. **机器人重启后必须重跑 `start_lidar.sh`**——LiDAR 默认不发数据，需要每次开机后用 driver 唤醒一次。
2. **同时只能有一个主机收数据**——`host_net_info` 是单值的。如果想多机同时看，要么改成 multicast，要么再加一个 forwarder。
3. **激光在镜面/玻璃/暗色织物上反射弱**——会出现误检或漏检，尤其在 5m+ 范围。穿亮色衣服测障碍物检测更稳。
4. **G1 机身会遮挡部分 FOV**——LiDAR 装在机器人顶部，但下方仍能看到自己的肩臂。这就是为什么我们必须设 `MIN_R=0.3`，不然"最近障碍"永远是机器人自己。

---

## 8) 下一步

把这套从"看到障碍物"升级到"避开障碍物"：
- 现成实现：`deploy_dodge_real.py` 的 `LidarObstacleDetector` + `DodgeController`
- 训练好的 dodge policy 用 18 维观测：自身姿态 + 障碍物位置/速度 → 输出速度命令
- sim2sim 验证（不需要真机）：`uv run python deploy_dodge_mujoco.py`

但**先用 `nearest_obstacle.py` 把传感器调好**——别带着没校准的 LiDAR 进策略。
