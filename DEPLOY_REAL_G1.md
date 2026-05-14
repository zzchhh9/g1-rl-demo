# G1 真机部署指南：Dodge+Return 策略

## 前置条件

- Unitree G1 29DOF (EDU 版，带 Jetson Orin NX)
- 头顶 Livox Mid-360 LiDAR 已安装
- 遥控器可用（用于启动/停止）
- 工作站已连接到 G1 的网络

---

## 第一步：工作站环境准备

```bash
# 1. 克隆你的 fork
git clone https://github.com/zzchhh9/g1-rl-demo.git
cd g1-rl-demo
git submodule update --init --recursive

# 2. 安装依赖 (uv)
uv sync

# 3. 确认 sim2sim 能跑（先不接真机）
MUJOCO_GL=egl PYTHONPATH="third_party/unitree_rl_gym:third_party/rsl_rl" \
  python deploy_dodge_mujoco.py --duration 10

# 看到 "✓ IDLE → DODGE → RETURN → STOP" 就说明环境没问题
```

## 第二步：sim2sim 验证（必须通过再接真机）

```bash
# 录制视频确认无碰撞
MUJOCO_GL=egl PYTHONPATH="third_party/unitree_rl_gym:third_party/rsl_rl" \
  python deploy_dodge_mujoco.py --record sim2sim_test.mp4 --duration 25

# 看视频确认：
# ✅ 机器人站稳不倒
# ✅ 障碍物靠近时机器人侧移躲避
# ✅ 障碍物远离后机器人回到原位
# ✅ 全程无碰撞
```

## 第三步：检查真机通信

```bash
# 1. 确认网络连接（G1 默认 IP 通常是 192.168.123.161）
ping 192.168.123.161

# 2. 测试 SDK 通信
python -c "
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
ChannelFactoryInitialize(0, 'eth0')  # 替换成你的网卡名
print('SDK 通信初始化成功')
"

# 如果报错：
# - 检查网卡名：ip link show
# - 检查防火墙：sudo ufw status
# - 确认 G1 已开机且在同一子网
```

## 第四步：单独测试 locomotion（不带 dodge）

**这一步非常重要 — 先确认 locomotion 能正常走路，再加 dodge。**

```bash
# 用官方的 deploy_real.py 测试基本行走
cd third_party/unitree_rl_gym/deploy/deploy_real
python deploy_real.py eth0 configs/g1.yaml

# 操作流程：
# 1. 遥控器按 START 键 → 退出零力矩状态
# 2. 机器人自动移到默认站姿
# 3. 按 A 键 → 开始行走
# 4. 用左摇杆控制：上=前进，下=后退，左右=侧移
# 5. 右摇杆左右 = 转向
# 6. 按 SELECT 键 → 安全停止

# 验证清单：
# ✅ 机器人能站稳
# ✅ 前进 0.3 m/s 稳定
# ✅ 侧移 0.3 m/s 稳定（最关键 — dodge 需要侧移）
# ✅ 转向 0.5 rad/s 稳定
# ✅ 停止命令后机器人停下不倒
```

## 第五步：LiDAR 验证

```bash
# 1. 启动 Livox 驱动（G1 上或工作站上）
# 如果用 ROS2:
ros2 launch livox_ros_driver2 msg_MID360_launch.py

# 2. 查看点云
ros2 topic echo /livox/lidar --once
# 确认有 PointCloud2 数据

# 3. 测试障碍物检测
# 让人站在机器人前方 2m 处
# 在工作站运行：
python -c "
# 简单测试：读取 LiDAR 点云，过滤地面，找最近障碍物
import rclpy
from sensor_msgs.msg import PointCloud2
import numpy as np

# ... (需要根据实际 ROS2 setup 调整)
# 关键：确认能得到障碍物的 [x, y] 坐标（body frame）
"

# 验证清单：
# ✅ LiDAR 有数据输出
# ✅ 能检测到 2m 外的人
# ✅ 位置精度 < 10cm
# ✅ 更新频率 ~10 Hz
```

## 第六步：Dodge 策略集成（低速测试）

**第一次测试务必用低速！**

需要修改 `deploy_real.py` 添加 dodge 逻辑。核心改动：

```python
# 在 Controller.run() 中，替换遥控器命令为 dodge 策略输出：

# 原来：
self.cmd[0] = self.remote_controller.ly      # 遥控器前后
self.cmd[1] = self.remote_controller.lx * -1  # 遥控器左右
self.cmd[2] = self.remote_controller.rx * -1  # 遥控器转向

# 改为：
if dodge_active:
    # dodge 策略输出速度命令
    self.cmd[0] = vel_cmd[0]  # vx from dodge policy
    self.cmd[1] = vel_cmd[1]  # vy from dodge policy  
    self.cmd[2] = vel_cmd[2]  # vrz from dodge/return
else:
    # 正常遥控器控制
    self.cmd[0] = self.remote_controller.ly
    self.cmd[1] = self.remote_controller.lx * -1
    self.cmd[2] = self.remote_controller.rx * -1
```

**第一次测试参数（保守）：**

```python
MAX_LIN_VEL = 0.15   # 只给 0.15 m/s（正常是 0.5）
MAX_ANG_VEL = 0.3    # 只给 0.3 rad/s（正常是 1.0）
SAFETY_DISTANCE = 2.0 # 2m 就开始躲（留足反应时间）
```

## 第七步：逐步提速

每一步都要确认安全后再进下一步：

| 轮次 | MAX_LIN_VEL | 障碍物速度 | 预期效果 |
|---|---|---|---|
| 1 | 0.15 m/s | 人慢走 0.1 m/s | 机器人缓慢侧移 |
| 2 | 0.25 m/s | 人正常走 0.2 m/s | 机器人明显侧移 |
| 3 | 0.35 m/s | 人正常走 0.3 m/s | 接近训练速度 |
| 4 | 0.50 m/s | 人快走 0.3 m/s | 全速（训练配置） |

---

## 安全检查清单

每次测试前确认：

- [ ] 急停按钮在手边
- [ ] 工作区域清空（无杂物、无其他人）
- [ ] 机器人电量 > 50%
- [ ] 遥控器 SELECT 键测试过可以停止
- [ ] LiDAR 数据正常
- [ ] 先跑 30 秒纯站立确认稳定

**急停条件（立即按 SELECT）：**
- 机器人明显倾斜 > 15°
- 机器人向障碍物方向移动（应该远离）
- 任何异常抖动或震荡
- LiDAR 丢失超过 1 秒

---

## 故障排除

### 机器人不动
- 检查 `cmd` 是否正确传递给 locomotion obs
- `cmd_scale` 要匹配 config：`[2.0, 2.0, 0.25]`
- 确认 dodge policy 输出不是全零（打印 `vel_cmd`）

### 机器人动了但方向反
- 检查 body frame 坐标系：X=前，Y=左
- IMU 四元数格式：`[w, x, y, z]`（不是 `[x, y, z, w]`）
- LiDAR 数据可能是 sensor frame，需要转换到 body frame

### 机器人摔倒
- 降低 MAX_LIN_VEL（先用 0.1 m/s）
- 检查 locomotion policy 是否正常（第四步要通过）
- 检查 PD 增益是否和 sim 一致

### LiDAR 检测不到障碍物
- Livox 驱动是否启动
- 点云 topic 是否有数据
- 地面过滤阈值是否太高（默认 Z > 0.3m）
- 人可能穿了低反射率的衣服 — 换亮色衣服

### 回到原位不准
- 里程计漂移 — 短时间（< 20s）内应该 < 10cm
- yaw 漂移 — P 控制器增益 kp=2.0，可以调到 3.0
- 如果漂移严重，考虑用 VIO 替代 SDK 里程计

---

## 文件清单

| 文件 | 用途 | 在哪里运行 |
|---|---|---|
| `deploy_dodge_mujoco.py` | sim2sim 验证 | 开发机 |
| `third_party/.../deploy_real.py` | 真机 locomotion | 工作站（连G1） |
| `deploy/dodge_policy.py` | dodge 策略推理 | 工作站 |
| `deploy/lidar_sim.py` | LiDAR 仿真（sim2sim 用） | 开发机 |
| `checkpoints/return_head_v23b_v6.pt` | 回位策略 | 工作站 |
| `policy_lstm_1.pt` / `motion.pt` | locomotion 策略 | 工作站 |

---

## obs 维度速查

### Dodge Policy (18 dim)
```
[0:3]   机器人速度 (body frame)      ← IMU
[3:6]   障碍物位置 (body frame)      ← LiDAR
[6:9]   障碍物速度 (body frame)      ← LiDAR 帧间差分
[9:11]  位移 (body frame, 从dodge开始) ← 里程计
[11:14] 静态障碍物 (可选, 默认0)      ← 硬编码
[14]    yaw 偏移 (从dodge开始)        ← IMU
[15:18] 上一步动作                    ← 内部状态
```

### Locomotion Policy (47 dim)
```
[0:3]   角速度 × 0.25               ← IMU gyro
[3:6]   重力方向                     ← IMU quat
[6:9]   速度命令 × cmd_scale         ← dodge policy 输出
[9:21]  关节位置 (相对默认)           ← 电机编码器
[21:33] 关节速度 × 0.05             ← 电机编码器
[33:45] 上一步动作                   ← 内部状态
[45:47] sin/cos 步态相位             ← 内部计时器
```
