# G1 Camera Dump — 2026-05-15

机器人关机前从 PC1 (Jetson Orin NX, 192.168.123.164) 抓的全部相机相关信息。

---

## 硬件结论

| 项 | 值 |
|---|---|
| 机器人计算单元 | NVIDIA Orin NX Developer Kit (Jetson) |
| 相机设备节点 | **`/dev/video4`** （USB-UVC，1920x1080 YUY2 @ 15 fps） |
| 当前在不在线 | **NO** — `/dev/video*` 都不存在；`lsusb -t` 只有 Wi-Fi 适配器，没相机。可能拔了或被关掉了 |
| 内参/外参 | **没找到任何标定文件** — 需要自己用棋盘格 + opencv 标定 |
| 历史用过 RealSense | 是（2023.11 的 ros 日志里有 `realsense2_camera` 启动记录），但当前没插 |
| 也有 VIO workspace | `/home/unitree/unitree/Odometer_service/`（基于 SVO Pro），里面只有 SVO benchmark 数据集的 demo 标定，**不是 G1 自身的相机标定** |

---

## 相机数据通路

机器人侧服务名：`video_hub_pc4`（DDS 域 0，eth0）
二进制：`/unitree/module/video_hub_pc4/videohub_pc4`（aarch64 ELF，63KB，本目录有 `videohub_pc4.bin`）
源文件路径（编译时）：`/home/unitree/sjy/g1_videohub_nx/videohub.c`（在机器人 root home，未抓——非 root 不可读）

### GStreamer pipeline（从 strings 抽出来）

```
v4l2src device=/dev/video4 
  ! video/x-raw, format=YUY2, width=1920, height=1080, framerate=15/1
  ! queue ! nvvidconv ! video/x-raw(memory:NVMM), format=NV12 
  ! tee name=vid 

  ! queue ! nvvidconv ! video/x-raw(memory:NVMM), format=NV12, width=1280, height=720 
  ! queue ! nvv4l2h264enc bitrate=8000000 iframeinterval=15 idrinterval=15 insert-sps-pps=1 
  ! tee name=enc ! queue ! appsink name=stream720p

  vid. ! queue ! nvjpegenc ! queue ! appsink name=image1080p

  vid. ! queue ! nvvidconv ! video/x-raw(memory:NVMM), format=NV12, width=640, height=360 
  ! queue ! nvv4l2h264enc bitrate=800000 iframeinterval=15 idrinterval=15 insert-sps-pps=1 
  ! appsink name=stream360p
```

另外有一个 RTP/H.264 多播旁路（不一定开）：
```
... ! rtph264pay ! udpsink host=230.1.1.1 port=1720 multicast-iface=eth0 sync=false
```

### DDS Topics（在 domain 0 上）

| Topic | 类型 | 用途 |
|---|---|---|
| `rt/frontvideostream` | `unitree_go::msg::dds_::Go2FrontVideoData_` | 持续推送的 H.264 视频流（三个分辨率打包在一条 msg 里）|
| `rt/videohub/inner` | std_msgs::String | 内部状态 |
| `rt/api/videohub/request` | unitree_api::Request | RPC 请求 |
| `rt/api/videohub/response` | unitree_api::Response | RPC 响应 |

### Go2FrontVideoData 消息结构（`Go2FrontVideoData.idl`、`.msg`、`.hpp` 都有）

```idl
struct Go2FrontVideoData {
    uint64       time_frame;
    sequence<uint8> video720p;   // H.264 1280x720 编码后的字节流
    sequence<uint8> video360p;   // H.264 640x360 编码后的字节流
    sequence<uint8> video180p;   // H.264 320x180 编码后的字节流 (推测)
};
```

注意：三个字段都是 **H.264 编码后的 NALU**，不是 raw frame。要拿到 RGB 还得用 ffmpeg / PyAV 解码。

### RPC API（更好用的路径，可以拉 JPEG 帧）

服务 `videohub`，从 `video_api.hpp`:

| API ID | 名字 | 返回 |
|---|---|---|
| **1001** | `GetImageSample` | **JPEG 1080p 字节**（直接 PIL 解码就是 RGB）|

Cpp 用法（从 `video_client_example.cpp`）：

```cpp
unitree::robot::ChannelFactory::Instance()->Init(0);
unitree::robot::go2::VideoClient video_client;
video_client.SetTimeout(1.0f);
video_client.Init();

std::vector<uint8_t> jpeg_bytes;
int ret = video_client.GetImageSample(jpeg_bytes);
// ret == 0 表示成功，jpeg_bytes 就是 JPEG 字节流
```

**这是最简单的拉 RGB 帧路径**，不需要 H.264 解码。Python 等价实现：参考 `unitree_sdk2py` 仓库里的 `RobotClient` 模式，发请求到 `rt/api/videohub/request`、API ID 1001、拿 response 里的 binary data → PIL Image。

---

## 标定（**MISSING — 需要自己做**）

我们没找到任何相机内参/外参文件。要做 LiDAR-相机融合就必须：

1. **内参**：标准棋盘格 + `cv2.calibrateCamera()`
   - 印一张 7×9 黑白棋盘（25mm 方格）
   - 从 G1 摄像头拍 20+ 张不同角度图
   - 跑 `cv2.calibrateCamera()` 得到 K 矩阵 (fx, fy, cx, cy) + distortion
2. **外参（LiDAR → 相机的 TF）**：把一个能被两个传感器同时看见的标定物（黑白棋盘 + 反光胶带）放视野中，分别在相机像素坐标和 LiDAR xyz 中标注同一个点，PnP 求解
   - 或者用 `kalibr_calibrate_cameras` + `kalibr_calibrate_imu_camera`
3. **2D bbox → 3D 锥**：用内参把 YOLO 的 bbox 像素角点反投影成相机坐标系射线，再用外参变换到 LiDAR 坐标系

---

## 本目录文件清单

| 文件 | 说明 |
|---|---|
| `README.md` | 本文 |
| `00_probe_output.txt` | 全部探查输出（lsusb / lsmod / find / strings 等等）|
| `videohub_pc4.bin` | 机器人侧 videohub 二进制（aarch64 ELF）|
| `videohub_pc4.strings.txt` | 上面 binary 的所有 strings（10KB，含完整 GStreamer pipeline）|
| `videohub_cyclonedds.xml` | videohub 服务用的 DDS 网卡配置 |
| `video_hub_pc4_module.json` | videohub 安装包元数据 |
| `master_service__video_hub_pc4` | 服务启停命令定义 |
| `Go2FrontVideoData.idl` | 视频消息 IDL 定义 |
| `Go2FrontVideoData.msg` | ROS msg 形式 |
| `Go2FrontVideoData_.hpp` | C++ struct + DDS type traits |
| `go2_front_video_data_python.py` | rosidl 生成的 Python 类定义（参考）|
| `video_client.hpp` / `video_api.hpp` / `video_error.hpp` | Go2 VideoClient 完整 C++ API |
| `video_client_example.cpp` | Unitree 官方 GetImageSample 示例 |
| `b2_front_video_*.hpp` | B2 机型对应接口（结构相同，可选参考）|

---

## 下一步开发思路（机器人关机后笔记本上做的）

可以并行做的事，不依赖机器人在线：

1. **看 unitree_sdk2py 里有没有 VideoClient 的 Python 实现**。如果有，直接用。如果没有：
   - 用 `ChannelPublisher("rt/api/videohub/request", Request)` + `ChannelSubscriber("rt/api/videohub/response", Response)` 自己包装
   - API ID = 1001
2. **写 RGB 帧 grab 函数**：返回 numpy HxWx3 uint8
3. **YOLO v8 person detection**：`pip install ultralytics`，预训练权重，输入 RGB → 输出 bbox 列表
4. **印棋盘格 + 拍标定图**（重新开机时做）→ 跑 `cv2.calibrateCamera()` 得 K
5. **LiDAR↔相机外参**：找 1-2 个能同时被两传感器看到的标定点（角点 + 反光胶带），手动标 → solvePnP
6. **融合**：每帧 (LiDAR 点云, RGB) 同步取 → YOLO 出 bbox → bbox 像素角点反投影成锥 → 用锥过滤 LiDAR 点 → 取这些点的中心作为"人的 3D 位置"

YOLO 在你笔记本上跑得动（一张 GPU 30+ FPS），不用部署到 Jetson。

---

最后更新：2026-05-15，机器人关机前最后一次同步。
