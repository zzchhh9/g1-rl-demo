# Livox Mid-360 LiDAR 是怎么工作的

把 G1 头顶那个圆盘的 LiDAR 从硬件到协议拆开，捋一遍数据是怎么从一束激光变成 Python `numpy` 数组的。

---

## 1. 硬件层面：非重复扫描

Livox Mid-360 是一颗 **机械式 + 棱镜旋转** 的固态混合激光雷达。它 **不是**传统的多线 LiDAR（Velodyne 那种 16/32/64 线的）：

- **传统多线 LiDAR**：N 个发射器水平旋转，每圈打出 N 条平行的"环"，扫描图样**周期性重复**，分辨率 = N×水平点数。
- **Mid-360**：内部一个旋转棱镜把单个激光束折射成**非重复（non-repetitive）的玫瑰花图案**。短时间内点云稀疏但分布均匀，**累积越久越密**——理论上无限时间能达到接近全视场（FOV 360°×59°）100% 覆盖。

为什么这么设计？
- 成本低（单一激光器+棱镜，不像 64 线要 64 套激光发射/接收）
- 远距离精度好（功率全用在一束上）
- 但**单帧点云稀疏**，下游用法要相应改：要么累积多帧、要么用专门的算法（FAST-LIO, Livox-mapping 等）

工作参数（来自 Livox 官方）：
- 测距范围：0.1m – 70m
- 点率：**~200,000 点/秒**
- 视场：360°（水平） × 59° (-7° ~ +52°，垂直)
- 扫描帧率：10 Hz（"帧"指一次完整 UDP 包流的命名周期，不是几何意义上的扫完一圈）

---

## 2. 物理输出：UDP 数据包

LiDAR 用 **UDP** 把点云推给主机。**不是 Livox SDK 的串行接口，不是 ROS topic，就是裸 UDP**。SDK / ROS driver 只是这些 UDP 包的包装。

5 条独立的 UDP 流（端口固定）：

| LiDAR 发送端口 | 主机接收端口 | 内容 | 频率 |
|---|---|---|---|
| `56100` | `56101` | 控制命令（双向） | 按需 |
| `56200` | `56201` | **状态推送**（heartbeat + 错误码 + 配置） | ~1 Hz |
| `56300` | `56301` | **点云数据** | **~2 kHz** |
| `56400` | `56401` | IMU 数据 | ~200 Hz |
| `56500` | `56501` | 日志 | 按需 |

调试时的一个关键事实：**LiDAR 上电后默认不发任何点云**，只发 heartbeat。要它开始推点云，主机必须先发"set host info + enable point send"命令。这就是为什么我们最初在 `:56301` 听了好久什么都没收到——LiDAR 在等握手。

---

## 3. 网络握手：Livox SDK 2 协议

每个 UDP 包都遵循统一的帧格式：

```
┌─────────┬──────────────────────────────────────┬──────────────┐
│ offset  │ field                                │ size (bytes) │
├─────────┼──────────────────────────────────────┼──────────────┤
│   0     │ sof                = 0xAA            │      1       │
│   1     │ version            = 0x00            │      1       │
│   2     │ length             (total frame LE)  │      2       │
│   4     │ seq_num            (uint32 LE)       │      4       │
│   8     │ cmd_id             (uint16 LE)       │      2       │
│  10     │ cmd_type           0=REQ, 1=ACK      │      1       │
│  11     │ sender_type        0=host, 1=lidar   │      1       │
│  12     │ reserved                             │      6       │
│  18     │ crc16              (CRC-16/CCITT-FALSE 覆盖 0..17) │ 2 │
│  20     │ crc32              (zlib CRC-32 覆盖 payload)      │ 4 │
│  24     │ payload                              │   length-24  │
└─────────┴──────────────────────────────────────┴──────────────┘
```

主要的 `cmd_id`：

| cmd_id | 含义 |
|---|---|
| `0x0000` | LidarSearch — 主机广播探测 / LiDAR 广播 ACK |
| `0x0100` | **Set Parameters** — 配置 LiDAR（写一组 key-value） |
| `0x0101` | Get Parameters — 读 LiDAR 当前配置 |
| `0x0102` | **Lidar Push** — LiDAR 主动推送状态（heartbeat） |

`0x0100` 的 payload：

```
key_num (2B) | rsvd (2B) | [key (2B) | length (2B) | value (length B)] × N
```

关键 key：

| key | 名称 | value 长度 | 内容 |
|---|---|---|---|
| `0x0003` | PointSendEn | 1 | `0x01` 启用点云推送 |
| `0x0005` | StateInfoHostIpCfg | 8 | 4B 目标 IP + 2B dst port + 2B src port |
| `0x0006` | **LidarPointDataHostIpCfg** | 8 | 同上：告诉 LiDAR 把点云发到哪 |
| `0x0007` | LidarImuHostIpCfg | 8 | 同上 (IMU) |
| `0x0008` | CtlHostIpCfg | 8 | 同上 (命令 ACK) |

**典型握手序列**（livox_ros_driver2 干的事）：

```
host → lidar:56100   SET PARAMETERS { PointSendEn=1, *HostIpCfg=<host>:<port> }
lidar → host:56101   ACK ret_code=0 (成功)
lidar → host:56201   PUSH heartbeat (持续 1Hz)
lidar → host:56301   POINT CLOUD (持续 ~2kHz)
lidar → host:56401   IMU (持续 ~200Hz)
```

我们在 G1 上的实际情况：
1. LiDAR 已经在多播 `224.1.1.5:56201` 广播 heartbeat（说明之前被配置过 multicast 模式）
2. 但点云端口完全沉默 — `PointSendEn` 没开
3. 手撸 Python 发 SET 命令，LiDAR 沉默（我们的协议实现还差细节，比如 ctl_host 没设对，ACK 回到了过期 IP）
4. **最终用 SDK 编译好的 `livox_ros_driver2_node` 二进制干这件事**——SDK 实现完全正确

---

## 4. 点云包的内部结构

点云走 UDP 56300→56301。每个 UDP 包是一段"以太网包"（Livox 自家定义，不要跟物理以太网帧混淆）：

```c
struct LivoxLidarEthernetPacket {
    uint8_t  version;            // = 0
    uint16_t length;             // 整个 UDP payload 长度
    uint16_t time_interval;      // 单位 0.1 µs
    uint16_t dot_num;            // 本包内点数
    uint16_t udp_cnt;            // UDP 包序号
    uint8_t  frame_cnt;          // 帧号
    uint8_t  data_type;          // 见下表
    uint8_t  time_type;
    uint8_t  rsvd[12];
    uint32_t crc32;
    uint8_t  timestamp[8];
    uint8_t  data[];             // dot_num 个点
};
```

固定头部 **36 字节**，然后是 `dot_num` 个点。每个点的字节布局取决于 `data_type`：

| data_type | 含义 | 单点格式 | 字节/点 |
|---|---|---|---|
| 0 | IMU | float×6 (gyro xyz + acc xyz) | 24 |
| **1** | **CartesianHigh** | int32×3 (x,y,z, **单位 mm**) + uint8 refl + uint8 tag | **14** |
| 2 | CartesianLow | int16×3 (x,y,z, **单位 cm**) + refl + tag | 8 |
| 3 | Spherical | uint32 depth + uint16 θ + uint16 φ + refl + tag | 10 |

我们这套用 `data_type=1`（最高精度，毫米级）。每个 UDP 包内点数实测约 **96 点**（典型情况），到达频率约 2 kHz，所以总点率 ≈ 192k pts/s ≈ 200k pts/s（符合 Livox 标称）。

Python 解析就 7 行：

```python
ver, length, _ti, dot_num, _uc, _fc, dtype, _tt = struct.unpack_from('<BHHHHBBB', data, 0)
assert dtype == 1
points = data[36:]
for i in range(dot_num):
    x, y, z, refl, tag = struct.unpack_from('<iiiBB', points, i * 14)
    yield (x * 0.001, y * 0.001, z * 0.001)  # 转米
```

---

## 5. 在我们 G1 上的实际数据通路

设计上 LiDAR 应该插在 G1 板载 Jetson Orin NX（PC1，`192.168.123.164`）上，由 Unitree 自家的"utlidar"感知服务做"Livox UDP → DDS topic `rt/utlidar/voxel_map`"的桥接。

实际情况（这套是从零调出来的，踩坑非常多）：
- ✅ Mid-360 在 `192.168.123.120`，硬件正常
- ✅ Livox SDK + livox_ros_driver2 编译产物在 `/upgradePythonServer/temp/.../graph_pid_ws/` 已编译
- ❌ Unitree 的"utlidar"桥接服务**根本不存在**——`rt/utlidar/*` 这个 topic 名是 Go2 的协议，G1 上从未实现
- ❌ 上述 workspace 被上传到 `/upgradePythonServer/temp/` 但安装步骤（移到 `/unitree/module/`）从未执行
- ❌ Jetson 上 ros2 在 DDS domain 0 上初始化时 `bad_alloc`——因为 Unitree 自家 60+ 个 DDS topic 把 cyclonedds RMW 撑爆

**我们采取的实用路径**（不动任何机器人持久状态）：

```
┌────────────────┐  UDP @ 2kHz  ┌────────────────────────┐  UDP   ┌────────────────┐
│  Mid-360 .120  │ ───────────► │ Robot Jetson .164      │ ─────► │  Laptop .222   │
│                │              │                        │        │                │
│  data_type=1   │              │  (a) one-time:         │        │  test_lidar.py │
│  cartesian-mm  │              │      livox_ros_driver2 │        │  nearest_obs.. │
│                │              │      bootstraps stream │        │                │
│                │              │  (b) persistent:       │        │  parses Livox  │
│                │              │      ~30 line Python   │        │  UDP → numpy   │
│                │              │      udp forwarder     │        │                │
└────────────────┘              └────────────────────────┘        └────────────────┘
```

- 机器人侧：临时跑一次 driver（让 LiDAR 进入推流状态）→ kill driver → 启动 30 行 Python UDP 转发器。**所有文件都在 `/tmp/livox-run/`，重启即清。**
- 笔记本侧：监听本机 UDP `:56301`，按上面的 36 字节头 + 14 字节/点的格式 parse。

完整脚本：
- 启动：`scripts/start_lidar.sh`
- 停止：`scripts/stop_lidar.sh`
- 测试：`uv run python test_lidar.py`
- 实时距离：`uv run python nearest_obstacle.py`

---

## 6. 为什么坐标轴是这样的

`extrinsic_parameter` 在 `MID360_config_fixed.json` 里：

```json
"extrinsic_parameter": {
    "roll":  0.0,
    "pitch": 1.57079632,  // ≈ π/2
    "yaw":   3.14159265,  // ≈ π
    "x": 0, "y": 0, "z": 0
}
```

这个 extrinsic 应用在 LiDAR 自家坐标系上。LiDAR 自己的 native 系：
- z 轴 = 旋转棱镜的轴向（机械朝向)
- x/y 在水平面（但相对哪个方向不定）

`pitch=π/2, yaw=π` 把它转到我们用的右手系：
- **x = 机器人正前方**
- **+y = 机器人正右侧**（注意：这跟典型 ROS REP 105 的"+y=左"是相反的，livox_ros_driver2 应用上面的 yaw=π 后实际产出的就是右手反转的 y 轴）
- **z = 上方**

**实测验证**：人用一根防摔架吊住机器人，支撑架的左侧立柱在 LiDAR 视野里持续出现在 `xyz=(+0.57, -0.82, +1.25)`、bearing=-55°、水平距离 1.0m。立柱在物理上**在机器人左侧**，对应 y=-0.82 是负值——确认了"+y=右"。

bearing 角的物理含义：
- **bearing ≈ 0°**  → 机器人正前方
- **bearing > 0°**  → 机器人右侧（顺时针往后转）
- **bearing < 0°**  → 机器人左侧（逆时针往后转）
- **bearing ≈ ±180°** → 机器人正后方

如果用户在视场中**逆时针**绕机器人走（从顶视图看），bearing 应该 **从 0° 减小到 -180°、跨过 ±180° 跳到 +180°、再减小回 0°**（"减小"是 atan2 数学意义上的；物理上人是从前→左→后→右→前）。

---

## 7. 下一步参考阅读

- **如何用这些数据做障碍物检测**：[lidar_nearest_obstacle.md](lidar_nearest_obstacle.md)
- **官方协议文档**：[Livox-SDK2 wiki - Mid-360 eth protocol](https://github.com/Livox-SDK/livox_wiki_en/blob/master/source/tutorials/new_product/mid360/livox_eth_protocol_mid360.md)
- **官方 SDK 源码**：[Livox-SDK2](https://github.com/Livox-SDK/Livox-SDK2) | [livox_ros_driver2](https://github.com/Livox-SDK/livox_ros_driver2)
