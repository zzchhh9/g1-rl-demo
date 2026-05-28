# Experiment Index

Sequential numbering for the dodge / gated-return deployment runs.
Experiment #1 is the latest run as of 2026-05-28; each new run shared afterward
is appended with the next number (2, 3, 4, ...).

| # | Run dir | Date | Mode / config | Result |
|---|---------|------|---------------|--------|
| 1 | runs/rhea_20260528_233107 | 2026-05-28 23:32 | gated return; return_head fed GEO online-frame `disp_b`; analytic yaw (output override); state-machine gate; `lin_vel=0.35` | **RETURN DONE est_disp=0.09m** — stable recover, converged (no walk-away) |
| 2 | runs/rhea_20260528_235319 | 2026-05-28 23:54 | gated return; yaw now from return_head's OWN `v_rz` (input sign-corrected, `--return_gated_yaw_sign -1`), GEO-frame `disp_b`; `lin_vel=0.35` | **RETURN DONE est_disp=0.09m** — full learned 3-DOF `a_ret`; `ret[2]` tapered +0.50→+0.31, heading converged +0.61→+0.33 |
