"""测试 G1 头顶 Livox Mid-360 LiDAR 的输入输出。

通过 unitree_sdk2py 的 DDS 订阅 G1 板载 LiDAR topic，
不需要 ROS2，不需要 Livox SDK，直接用 Unitree SDK 自带的接口。

G1 板载 LiDAR 可用 topic:
    rt/utlidar/voxel_map            - 体素点云 (PointCloud2)
    rt/utlidar/voxel_map_compressed - 压缩版
    rt/utlidar/height_map           - 高度图 (PointCloud2)
    rt/utlidar/range_map            - 距离图 (PointCloud2)

Usage:
    # 连接 G1 测试 LiDAR
    uv run python test_lidar.py eth0

    # 指定 topic
    uv run python test_lidar.py eth0 --topic rt/utlidar/voxel_map

    # 测试障碍物检测（让人站在机器人前方 2m）
    uv run python test_lidar.py eth0 --detect

参考:
    - unitree_sdk2py: https://github.com/unitreerobotics/unitree_sdk2_python
    - Sentdex g1_vibes: https://github.com/Sentdex/unitree_g1_vibes
"""

from __future__ import annotations

import argparse
import struct
import sys
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_


# PointField datatype 常量 (sensor_msgs/PointField, 与 SDK PointField_Constants 一致)
INT8 = 1
UINT8 = 2
INT16 = 3
UINT16 = 4
INT32 = 5
UINT32 = 6
FLOAT32 = 7
FLOAT64 = 8


class LidarTest:
    """订阅 G1 LiDAR DDS topic，解析 PointCloud2 数据。"""

    def __init__(self, topic: str = "rt/utlidar/voxel_map"):
        self.topic = topic
        self._msg_count = 0
        self._latest_points: np.ndarray | None = None
        self._latest_time = 0.0

        self.subscriber = ChannelSubscriber(topic, PointCloud2_)
        self.subscriber.Init(self._callback, 10)
        print(f"[LiDAR] 已订阅 topic: {topic}")
        print(f"[LiDAR] 等待数据...")

    def _callback(self, msg: PointCloud2_):
        """解析 PointCloud2 消息，提取 xyz 坐标。"""
        self._msg_count += 1
        self._latest_time = time.time()

        width = msg.width
        height = msg.height
        point_step = msg.point_step
        data = bytes(msg.data)
        fields = msg.fields
        n_points = width * height

        if n_points == 0 or len(data) == 0:
            return

        # 找到 x, y, z 字段的 offset 和 datatype
        field_map = {}
        for f in fields:
            field_map[f.name] = (f.offset, f.datatype)

        if "x" not in field_map or "y" not in field_map or "z" not in field_map:
            if self._msg_count <= 3:
                print(f"[LiDAR] WARNING: 缺少 x/y/z 字段。"
                      f"可用字段: {list(field_map.keys())}")
            return

        x_off, x_type = field_map["x"]
        y_off, y_type = field_map["y"]
        z_off, z_type = field_map["z"]

        # 解析点云
        points = []
        for i in range(n_points):
            base = i * point_step
            if base + point_step > len(data):
                break
            x = self._read_field(data, base + x_off, x_type)
            y = self._read_field(data, base + y_off, y_type)
            z = self._read_field(data, base + z_off, z_type)
            if x is not None and y is not None and z is not None:
                points.append([x, y, z])

        if len(points) > 0:
            self._latest_points = np.array(points, dtype=np.float32)

    @staticmethod
    def _read_field(data: bytes, offset: int, datatype: int) -> float | None:
        """从 PointCloud2 data 中读取一个字段值。"""
        try:
            if datatype == FLOAT32:
                return struct.unpack_from("<f", data, offset)[0]
            elif datatype == FLOAT64:
                return struct.unpack_from("<d", data, offset)[0]
            elif datatype == INT32:
                return float(struct.unpack_from("<i", data, offset)[0])
            else:
                return None
        except struct.error:
            return None

    @property
    def points(self) -> np.ndarray | None:
        return self._latest_points

    @property
    def msg_count(self) -> int:
        return self._msg_count


def test_basic(lidar: LidarTest, duration: float = 10.0):
    """基础测试：检查 LiDAR 是否有数据输出。"""
    print(f"\n{'='*60}")
    print(f"测试 1: 基础连接测试 ({duration}s)")
    print(f"{'='*60}")

    start = time.time()
    while time.time() - start < duration:
        time.sleep(1.0)
        pts = lidar.points
        n_msgs = lidar.msg_count

        if pts is not None:
            n = len(pts)
            x_range = f"[{pts[:,0].min():.2f}, {pts[:,0].max():.2f}]"
            y_range = f"[{pts[:,1].min():.2f}, {pts[:,1].max():.2f}]"
            z_range = f"[{pts[:,2].min():.2f}, {pts[:,2].max():.2f}]"
            print(f"  收到 {n_msgs:4d} 帧 | 最新: {n:5d} 点 | "
                  f"X{x_range} Y{y_range} Z{z_range}")
        else:
            print(f"  收到 {n_msgs:4d} 帧 | 无点云数据")

    if lidar.msg_count == 0:
        print("\n❌ 失败: 没有收到任何 LiDAR 数据")
        print("   检查:")
        print("   1. G1 是否开机且 LiDAR 已启动")
        print("   2. 网络连接是否正常 (ping 192.168.123.161)")
        print("   3. DDS 域是否匹配")
        return False

    if lidar.points is None or len(lidar.points) == 0:
        print("\n⚠️  收到消息但无点云，可能 topic 格式不对")
        print(f"   尝试其他 topic: rt/utlidar/height_map")
        return False

    print(f"\n✅ 通过: 收到 {lidar.msg_count} 帧，最新帧 {len(lidar.points)} 点")
    return True


def test_frequency(lidar: LidarTest, duration: float = 5.0):
    """频率测试：检查 LiDAR 更新频率。"""
    print(f"\n{'='*60}")
    print(f"测试 2: 频率测试 ({duration}s)")
    print(f"{'='*60}")

    count_start = lidar.msg_count
    time.sleep(duration)
    count_end = lidar.msg_count

    freq = (count_end - count_start) / duration
    print(f"  帧数: {count_end - count_start} / {duration}s = {freq:.1f} Hz")

    if freq < 5:
        print(f"  ⚠️  频率偏低 (预期 ~10 Hz)")
    elif freq > 15:
        print(f"  ⚠️  频率偏高 (预期 ~10 Hz)")
    else:
        print(f"  ✅ 频率正常")

    return freq


def test_obstacle_detection(lidar: LidarTest, duration: float = 15.0):
    """障碍物检测测试：让人站在机器人前方，检测是否能识别。"""
    print(f"\n{'='*60}")
    print(f"测试 3: 障碍物检测 ({duration}s)")
    print(f"  请让一个人站在机器人前方 1-3m 处")
    print(f"{'='*60}")

    min_height = 0.3   # 过滤地面
    max_range = 5.0    # 最远检测距离

    for i in range(int(duration)):
        time.sleep(1.0)
        pts = lidar.points
        if pts is None or len(pts) == 0:
            print(f"  {i+1:2d}s: 无数据")
            continue

        # 过滤地面点 (z > min_height)
        above_floor = pts[pts[:, 2] > min_height]

        # 过滤远距离点
        dists = np.linalg.norm(above_floor[:, :2], axis=1)
        nearby = above_floor[(dists > 0.3) & (dists < max_range)]

        if len(nearby) == 0:
            print(f"  {i+1:2d}s: {len(pts):5d} 点 (地面上方: {len(above_floor)}, "
                  f"近距离: 0) — 未检测到障碍物")
            continue

        # 最近障碍物的质心
        centroid = nearby.mean(axis=0)
        nearest_dist = dists[(dists > 0.3) & (dists < max_range)].min()

        print(f"  {i+1:2d}s: {len(pts):5d} 点 | 障碍物: "
              f"pos=[{centroid[0]:+.2f}, {centroid[1]:+.2f}, {centroid[2]:.2f}] "
              f"最近={nearest_dist:.2f}m 点数={len(nearby)}")

    if lidar.points is not None:
        pts = lidar.points
        above = pts[pts[:, 2] > min_height]
        dists = np.linalg.norm(above[:, :2], axis=1)
        nearby = above[(dists > 0.3) & (dists < max_range)]
        if len(nearby) > 10:
            print(f"\n✅ 障碍物检测成功: {len(nearby)} 个点在 0.3-{max_range}m 范围内")
            return True

    print(f"\n⚠️  未能稳定检测到障碍物")
    print(f"   确认: 人是否站在 1-3m 范围内？LiDAR 是否被遮挡？")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G1 LiDAR 测试工具")
    parser.add_argument("net", type=str, help="网卡名 (如 eth0)")
    parser.add_argument("--topic", type=str, default="rt/utlidar/voxel_map",
                        help="LiDAR DDS topic (默认: rt/utlidar/voxel_map)")
    parser.add_argument("--detect", action="store_true",
                        help="运行障碍物检测测试")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="每项测试时长 (秒)")
    args = parser.parse_args()

    # 初始化 DDS 通信
    ChannelFactoryInitialize(0, args.net)
    print(f"[DDS] 已初始化，网卡: {args.net}")

    lidar = LidarTest(topic=args.topic)

    # 等待一下让订阅建立
    time.sleep(1.0)

    # 测试 1: 基础连接
    ok = test_basic(lidar, args.duration)
    if not ok:
        print("\n基础测试失败，跳过后续测试。")
        print("如果 topic 不对，尝试:")
        print("  uv run python test_lidar.py eth0 --topic rt/utlidar/height_map")
        print("  uv run python test_lidar.py eth0 --topic rt/utlidar/range_map")
        sys.exit(1)

    # 测试 2: 频率
    test_frequency(lidar, min(args.duration, 5.0))

    # 测试 3: 障碍物检测 (可选)
    if args.detect:
        test_obstacle_detection(lidar, args.duration)

    print(f"\n{'='*60}")
    print(f"测试完成。共收到 {lidar.msg_count} 帧。")
    print(f"{'='*60}")
