"""Read current arm + waist joint angles from G1 LowState DDS.

Robot must be ON, in any state. Doesn't send any motor commands.
"""
import sys, time
import numpy as np
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG

NET = sys.argv[1] if len(sys.argv) > 1 else "eno1"

# Indices and kps from configs/g1.yaml:
LEG_IDX = list(range(0, 12))                  # 0-11
ARM_WAIST_IDX = list(range(12, 29))           # 12-28 (3 waist + 7+7 arms)
NAMES = (
    [f"leg_{i}" for i in range(12)] +
    ["waist_yaw", "waist_roll", "waist_pitch",
     "L_shoulder_pitch", "L_shoulder_roll", "L_shoulder_yaw",
     "L_elbow",  "L_wrist_roll", "L_wrist_pitch", "L_wrist_yaw",
     "R_shoulder_pitch", "R_shoulder_roll", "R_shoulder_yaw",
     "R_elbow",  "R_wrist_roll", "R_wrist_pitch", "R_wrist_yaw"]
)
KPS = ([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40] +
       [300, 300, 300, 100, 100, 50, 50, 20, 20, 20,
                       100, 100, 50, 50, 20, 20, 20])
ARM_WAIST_TARGET_DEFAULT = [0.0] * 17

ChannelFactoryInitialize(0, NET)
state = {"q": None}
def cb(msg: LowStateHG):
    state["q"] = [float(msg.motor_state[i].q) for i in range(29)]
sub = ChannelSubscriber("rt/lowstate", LowStateHG)
sub.Init(cb, 10)

print(f"Waiting for lowstate on {NET}...")
t0 = time.time()
while state["q"] is None and time.time() - t0 < 5:
    time.sleep(0.1)
if state["q"] is None:
    print("ERR: no lowstate received in 5s — robot off?"); sys.exit(1)

q = state["q"]
print(f"\n{'name':<22} {'idx':>3} {'q (rad)':>10} {'q (deg)':>9} "
      f"{'target':>8} {'err(rad)':>9} {'kp':>4}  {'shock force (Nm)':>16}")
print("-" * 95)
# legs
for i in range(12):
    err = abs(q[i] - 0.0)
    print(f"{NAMES[i]:<22} {i:>3} {q[i]:+10.3f} {np.degrees(q[i]):+9.1f} {0.0:>8.2f} "
          f"{err:>9.3f} {KPS[i]:>4}  {KPS[i]*err:>16.1f}")
print("-" * 95)
# arm + waist
print("--- ARM/WAIST (these whip when start pressed) ---")
for i, motor_idx in enumerate(ARM_WAIST_IDX):
    target = ARM_WAIST_TARGET_DEFAULT[i]
    err = abs(q[motor_idx] - target)
    kp = KPS[motor_idx]
    flag = "  ← WHIP RISK" if kp * err > 30 else ""
    print(f"{NAMES[motor_idx]:<22} {motor_idx:>3} {q[motor_idx]:+10.3f} {np.degrees(q[motor_idx]):+9.1f} "
          f"{target:>8.2f} {err:>9.3f} {kp:>4}  {kp*err:>16.1f}{flag}")

print(f"\n>>> Suggested arm_waist_target to AVOID whip (just paste into g1.yaml):")
arm_q = [round(q[i], 4) for i in ARM_WAIST_IDX]
print(f"arm_waist_target: {arm_q}")
print(f"\n>>> Or temporarily reduce arm_waist_kps to:")
soft_kp = [max(5, k // 5) for k in KPS[12:29]]
print(f"arm_waist_kps: {soft_kp}")
