#!/usr/bin/env bash
# ============================================================================
# 45/50 同款 (run rhea_20260531_144452 = RETURN DONE 0.05m):
#   balance_mode=0, 无 rear_bias (policy 自由 dodge), external SLAM odom, geo return.
#
# 每个 run 恰好【1 次】dodge+recover (无 --loop).
#
# 本次调整: balance_mode=0 下前进比后退慢得多(return 实际 ~0.06m/s, 而后退能跟上
# 0.30), dodge 又 coast 到 1.2-1.45m -> 慢速 15s timeout 回不到. 提 return 上限+时间:
#   --return_max_vel 0.25->0.40  --return_timeout 15->30s  --return_no_progress 3->6s
# (return 有 P 减速+min_vel, 近原点仍慢下来收敛, 不影响最终精度.)
#
# 又: dodge 后 coast(惯性后退)太大(settle speed 0.4+m/s 降不下来) -> "did not
# settle", 或 settle 后 return 时机器人还在后退 -> "moving away" abort. 降 dodge
# 速度减小 coast:  MAX_VEL 0.30 -> 0.20
#
# fake YOLO 是一次性的(发完一轮就停), 所以每个 run 都【重起 fake YOLO】发一次.
#   - 之前的 bug: 复用 sensors 时连 fake YOLO 也跳过了重启 -> 只有第 1 个 run 有
#     假人能 dodge, 后续 run 全程 track=-1 不 dodge.
#   - 现在: 复用 lidar/SLAM(不重启、不卡 "[lidar-ros2] connecting"),
#     但【每次都单独重起 fake YOLO】发 1 次.
#
# 用法:
#   ./test.sh                 复用 sensors + 重发 1 次 fake YOLO -> 1 次 dodge+recover
#   FORCE_START=1 ./test.sh   完整重启整套 sensors (首次/odom 坏了/换环境)
# ============================================================================
set -u
cd "$(dirname "$0")"
NET="${NET:-eno1}"
ROOT="$PWD"

# 一次性 fake YOLO(无 --loop): idle 8s -> 正前方 0.5m hold -> fade 远离 -> 停
FY="cd '$ROOT' && uv run python scripts/fake_yolo_obstacle_dds.py '$NET' \
  --topic rt/yolo/person --start-delay 8 --bearing-deg 0 --track-id 9001; bash"

AUTO=1
if [ "${FORCE_START:-0}" != "1" ] \
   && tmux has-session -t g1-dodge 2>/dev/null \
   && timeout 8 uv run python scripts/dds_wait_json.py "$NET" --topic rt/dodge/odom \
        --stale 0.35 --timeout 5 --min-count 5 --stable-seconds 1.0 >/dev/null 2>&1; then
  AUTO=0
  echo "[test] 复用 lidar/SLAM(不碰 lidar) + 重起 fake YOLO 发 1 次"
  tmux respawn-window -k -t g1-dodge:yolo "$FY" 2>/dev/null \
    || tmux new-window -t g1-dodge -n yolo "$FY"
else
  echo "[test] 完整启动 sensors + fake YOLO 发 1 次(会有一次 lidar connecting)"
fi

exec env AUTO_START_SENSORS_BEFORE_DEPLOY="$AUTO" FAKE_YOLO_START_DELAY=8 \
  MAX_VEL=0.20 \
  ./deploy_fake.sh \
  --balance_mode 0 --no_require_loco_ready \
  --return_mode geo --return_done_dist 0.06 \
  --return_max_vel 0.40 --return_timeout 30 --return_no_progress_timeout 6 \
  "$@"
