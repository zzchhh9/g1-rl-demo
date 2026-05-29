# Experiment Index

Sequential numbering for the dodge / gated-return deployment runs.
Experiment #1 is the latest run as of 2026-05-28; each new run shared afterward
is appended with the next number (2, 3, 4, ...).

| # | Run dir | Date | Mode / config | Result |
|---|---------|------|---------------|--------|
| 1 | runs/rhea_20260528_233107 | 2026-05-28 23:32 | gated return; return_head fed GEO online-frame `disp_b`; analytic yaw (output override); state-machine gate; `lin_vel=0.35` | **RETURN DONE est_disp=0.09m** — stable recover, converged (no walk-away) |
| 2 | runs/rhea_20260528_235319 | 2026-05-28 23:54 | gated return; yaw now from return_head's OWN `v_rz` (input sign-corrected, `--return_gated_yaw_sign -1`), GEO-frame `disp_b`; `lin_vel=0.35` | **RETURN DONE est_disp=0.09m** — full learned 3-DOF `a_ret`; `ret[2]` tapered +0.50→+0.31, heading converged +0.61→+0.33 |
| 3 | runs/rhea_20260529_001355 | 2026-05-29 00:13 | real-person (`YOLO_SOURCE=real`), gated; `--no_exit_on_return_abort` | **FAILED — no dodge.** Camera TCP stream (robot:5005) dropped `EOF` after ~5s, YOLO process exited → `track=-1` all 363 frames, 0 dodge. Robot then drifted ~1.5m+60° uncommanded in long idle. Odom DDS stayed fresh. |
| 4 | runs/rhea_20260529_003759 | 2026-05-29 00:37 | real-person, gated; first run with idle re-StopMove + YOLO reconnect | **Partial.** YOLO worked (dodges fired, track=11/19/24/35). But idle re-StopMove **failed to hold** — drifted to ~1.9m despite re-StopMove every tick (lio+low yaw both rotated ~1.5rad = real fsm=200 idle creep, free-standing). One RETURN ABORT. → StopMove can't pin the gait; need active position-hold. |
| 5 | runs/rhea_20260529_011112 | 2026-05-29 01:11 | real-person, gated, **`KEEP_VIDEOHUB=1`** (do NOT pkill Unitree video_hub) | **Root cause confirmed.** Camera worked (dodge, `RETURN DONE 0.10m`). Idle drift **bounded, no runaway**: pre-dodge plateaus ~0.5m, post-return ~0.2m (vs #4's 1.9m runaway). Only var changed = suppressing the video_hub/master_service kill ⇒ that kill was disturbing the loco service. Camera opens fine with video_hub alive ⇒ kill unnecessary. Fix: `KEEP_VIDEOHUB=1` now default. |
