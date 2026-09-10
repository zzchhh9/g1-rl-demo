# Docs index

操作入口是仓库根目录 [README.md](../README.md)。这里是更细的笔记，不是互相覆盖的多份「唯一教程」。

## 当前真机路径

| 文档 | 用途 |
| --- | --- |
| [g1_sdk_loco_dodge_runbook.md](g1_sdk_loco_dodge_runbook.md) | SDK 高层 loco dodge：蓝灯模式、急停、传感器、YOLO、回位 |
| [mid360_lio_recover.md](mid360_lio_recover.md) | Mid-360 → LIO → UDP → DDS `rt/dodge/odom` → 回位；验收与故障 |

栈默认 SLAM 是 **FAST-LIO**（`SLAM_BACKEND=fast_lio`，`/Odometry`）。LIO-SAM 仍可用（`SLAM_BACKEND=lio_sam`），且是 `./start_lidar_mapping.sh` 的建图后端。

## LiDAR 点云（非 LIO）

| 文档 | 用途 |
| --- | --- |
| [lidar_how_it_works.md](lidar_how_it_works.md) | Mid-360 硬件、UDP 协议、`start_lidar.sh` 裸转发 |
| [lidar_nearest_obstacle.md](lidar_nearest_obstacle.md) | `nearest_obstacle.py` 最近障碍实现 |

## 仓库根上的历史部署文

这些**不是**现在的默认操作手册，但有用：

- [DEPLOY_REAL_G1.md](../DEPLOY_REAL_G1.md) — 低层 `motion.pt`、sim2sim、早期 UDP LiDAR 工具
- [DEPLOY_PLAN.md](../DEPLOY_PLAN.md) — 2026-05 执行计划 / 门控清单
- [g1_camera_dump/README.md](../g1_camera_dump/README.md) — 相机 / RealSense / videohub 现场记录
