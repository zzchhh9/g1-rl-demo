"""单目相机 + YOLOv8 → 检测人 + 估算方位 / 距离 + 多帧跟踪.

支持两种模式：
    1) 单张图片：标注 bbox + 距离/方位 → 输出 annotated JPG
    2) 多帧目录或视频：逐帧检测 + ID 跟踪 + 轨迹图

距离估算（单目，无深度）：
    用 bbox 高度 + 假设人身高 1.7m + 相机内参（焦距），
    distance = (focal_length_px * person_height_meters) / bbox_height_px
    适用于:
      ✓ 人完整地站在画面中 (没被裁切)
      ✓ 人正面朝向相机 (侧身距离会偏大 ~20%)
      ✗ 坐着 / 蹲下 (bbox 高度变小 → 距离会被估远)

方位估算：
    bearing = atan2(bbox_cx - W/2, focal_length_px)  (deg, 正值=右)

相机内参（默认值对应 G1 头部 RealSense D435i RGB at 1280x720）：
    HFOV = 69° (Intel D435i 标称)
    focal_x_px = (W/2) / tan(HFOV/2) ≈ 932 px @ 1280

用法：
    # 单张
    uv run python person_tracker.py --input vision_demo/samples/bus.jpg

    # 全目录
    uv run python person_tracker.py --input vision_demo/samples/ --out vision_demo/outputs/

    # 自定义 FOV
    uv run python person_tracker.py --input X.jpg --hfov-deg 90

    # 改假设身高
    uv run python person_tracker.py --input X.jpg --person-height 1.65
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO


COCO_PERSON_CLASS = 0   # YOLO COCO class index for "person"


@dataclass
class PersonDetection:
    """单个人的检测结果，包含估算的几何量."""
    bbox_xyxy: tuple[int, int, int, int]   # x1, y1, x2, y2 in pixels
    confidence: float
    bearing_deg: float                       # +ve = right of center, deg
    elevation_deg: float                     # +ve = above center, deg
    distance_m: float                        # estimated by bbox height
    distance_from_width_m: float             # alt estimate from bbox width

    @property
    def cx(self): return (self.bbox_xyxy[0] + self.bbox_xyxy[2]) / 2
    @property
    def cy(self): return (self.bbox_xyxy[1] + self.bbox_xyxy[3]) / 2
    @property
    def w(self): return self.bbox_xyxy[2] - self.bbox_xyxy[0]
    @property
    def h(self): return self.bbox_xyxy[3] - self.bbox_xyxy[1]


class MonoCamera:
    """简单单目相机模型 — 仅小孔投影，无畸变."""

    def __init__(self, image_size_wh: tuple[int, int], hfov_deg: float):
        self.W, self.H = image_size_wh
        self.hfov = math.radians(hfov_deg)
        # focal length in pixels (assuming square pixels)
        self.fx = (self.W / 2) / math.tan(self.hfov / 2)
        self.fy = self.fx
        self.cx_px = self.W / 2
        self.cy_px = self.H / 2

    @property
    def vfov_deg(self):
        return math.degrees(2 * math.atan((self.H / 2) / self.fy))

    def bearing_from_x(self, x_px: float) -> float:
        return math.degrees(math.atan2(x_px - self.cx_px, self.fx))

    def elevation_from_y(self, y_px: float) -> float:
        # +ve elevation = camera looking up at point (pixel above center)
        # image y goes DOWN, so we negate
        return math.degrees(math.atan2(self.cy_px - y_px, self.fy))


def estimate_distance_from_height(bbox_h_px: float,
                                  person_height_m: float,
                                  focal_y_px: float) -> float:
    if bbox_h_px <= 1:
        return float("inf")
    return (focal_y_px * person_height_m) / bbox_h_px


def estimate_distance_from_width(bbox_w_px: float,
                                 person_width_m: float,
                                 focal_x_px: float) -> float:
    if bbox_w_px <= 1:
        return float("inf")
    return (focal_x_px * person_width_m) / bbox_w_px


def detect_persons(model: YOLO,
                   img_bgr: np.ndarray,
                   cam: MonoCamera,
                   person_height_m: float,
                   person_width_m: float,
                   conf_thresh: float = 0.30) -> List[PersonDetection]:
    """Run YOLO on one BGR image, return list of PersonDetection."""
    results = model.predict(img_bgr, classes=[COCO_PERSON_CLASS],
                            conf=conf_thresh, verbose=False, device='cpu')
    detections: List[PersonDetection] = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls.item())
            if cls != COCO_PERSON_CLASS:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf.item())
            cx = (x1 + x2) / 2
            # use mid-height of bbox for elevation (chest area)
            mid_y = (y1 + y2) / 2
            bearing = cam.bearing_from_x(cx)
            elev = cam.elevation_from_y(mid_y)
            dist_h = estimate_distance_from_height(y2 - y1, person_height_m, cam.fy)
            dist_w = estimate_distance_from_width(x2 - x1, person_width_m, cam.fx)
            detections.append(PersonDetection(
                bbox_xyxy=(int(x1), int(y1), int(x2), int(y2)),
                confidence=conf,
                bearing_deg=bearing,
                elevation_deg=elev,
                distance_m=dist_h,
                distance_from_width_m=dist_w,
            ))
    detections.sort(key=lambda d: d.distance_m)  # nearest first
    return detections


def annotate(img: np.ndarray, dets: List[PersonDetection], cam: MonoCamera) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    # vertical center line
    cv2.line(out, (w // 2, 0), (w // 2, h), (180, 180, 180), 1)
    # horizontal center line
    cv2.line(out, (0, h // 2), (w, h // 2), (180, 180, 180), 1)

    for i, d in enumerate(dets):
        x1, y1, x2, y2 = d.bbox_xyxy
        color = (0, 200, 0) if i == 0 else (0, 165, 255)   # nearest in green, others orange
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

        # label
        lines = [
            f"person {d.confidence:.2f}",
            f"d={d.distance_m:.2f}m (h)",
            f"  ={d.distance_from_width_m:.2f}m (w)",
            f"bearing={d.bearing_deg:+.1f}°",
        ]
        y0 = max(y1 - 6 - 18 * len(lines), 18)
        for k, txt in enumerate(lines):
            yy = y0 + k * 18
            cv2.putText(out, txt, (x1, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
            cv2.putText(out, txt, (x1, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        # crosshair on center
        cv2.circle(out, (int(d.cx), int(d.cy)), 4, color, -1)

    # bottom strip with camera info
    info = f"image {w}x{h}  HFOV={math.degrees(cam.hfov):.1f}°  VFOV={cam.vfov_deg:.1f}°  fx={cam.fx:.0f}px  persons={len(dets)}"
    cv2.rectangle(out, (0, h - 30), (w, h), (0, 0, 0), -1)
    cv2.putText(out, info, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return out


def plot_top_down(dets_per_frame: List[List[PersonDetection]],
                  out_path: Path,
                  max_range_m: float = 6.0,
                  hfov_deg: float = 69.0):
    """For each frame, plot the nearest person's (bearing,distance) → top-down trajectory."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    pts = []   # (frame_idx, x_right_m, y_forward_m)
    for i, dets in enumerate(dets_per_frame):
        if not dets:
            continue
        d = dets[0]   # nearest
        # convert (bearing, distance) → (x_right, y_forward) using HFOV-defined "robot frame"
        # +x = forward, +y = right (matching our LiDAR convention from earlier work)
        rad = math.radians(d.bearing_deg)
        x_fwd = d.distance_m * math.cos(rad)
        y_right = d.distance_m * math.sin(rad)
        pts.append((i, x_fwd, y_right))

    fig, ax = plt.subplots(figsize=(8, 8))
    PR = max_range_m + 0.5
    ax.set_xlim(-PR, PR); ax.set_ylim(-PR, PR)

    # front 180° shading (front = +x means up on plot)
    theta = np.linspace(-np.pi/2, np.pi/2, 120)
    ax.fill_between(max_range_m * np.sin(theta), 0, max_range_m * np.cos(theta),
                    color='lightblue', alpha=0.18)
    for r in [1, 2, 3, 4, 5]:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color='gray', ls='--', alpha=0.35))
        ax.text(0.06, r + 0.06, f"{r}m", fontsize=8, color='gray')

    # FOV cone in dashed line
    hfov_rad = math.radians(hfov_deg)
    for sign in (-1, 1):
        a = sign * hfov_rad / 2
        ax.plot([0, max_range_m * math.sin(a)], [0, max_range_m * math.cos(a)],
                color='steelblue', ls=':', alpha=0.6)

    # robot (camera) at origin
    ax.scatter([0], [0], s=400, c='black', marker='o', zorder=6, label='G1 camera')
    ax.annotate('', xy=(0, 0.6), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=2.5))

    if pts:
        idx, xs_fwd, ys_right = zip(*pts)
        # plot: matplotlib x = robot's y (right), matplotlib y = robot's x (forward)
        pp = np.array([ys_right, xs_fwd]).T.reshape(-1, 1, 2)
        if len(pp) > 1:
            segs = np.concatenate([pp[:-1], pp[1:]], axis=1)
            lc = LineCollection(segs, cmap='viridis', linewidth=2.5, alpha=0.85, zorder=3)
            lc.set_array(np.array(idx[:-1]))
            ax.add_collection(lc)
            cbar = plt.colorbar(lc, ax=ax, fraction=0.042, pad=0.04)
            cbar.set_label('frame index')
        ax.scatter(ys_right, xs_fwd, s=22, c='red', zorder=4)
        ax.scatter([ys_right[0]], [xs_fwd[0]], s=200, marker='o',
                   facecolor='lime', edgecolor='black', zorder=7, label=f'first (frame {idx[0]})')
        ax.scatter([ys_right[-1]], [xs_fwd[-1]], s=220, marker='X', c='crimson',
                   zorder=7, label=f'last (frame {idx[-1]})')

    ax.set_xlabel('y - camera right (m)  ->')
    ax.set_ylabel('x - camera forward (m)  ^')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_title(f'Person tracking (nearest person per frame)\n'
                 f'{len(pts)} valid frames / {len(dets_per_frame)} total  |  HFOV={hfov_deg:.1f}°')
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="YOLOv8 person detection + monocular distance / bearing")
    p.add_argument("--input", type=str, required=True, help="Image file or directory of images")
    p.add_argument("--out", type=str, default="vision_demo/outputs",
                   help="Output directory (annotated images)")
    p.add_argument("--model", type=str, default="yolov8n.pt",
                   help="YOLO model: yolov8n/s/m/l/x.pt (n = fastest, x = most accurate)")
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--hfov-deg", type=float, default=69.0,
                   help="Camera horizontal FOV (default 69° for D435i RGB)")
    p.add_argument("--person-height", type=float, default=1.70,
                   help="Assumed person height in meters (for distance estimate from bbox height)")
    p.add_argument("--person-width", type=float, default=0.45,
                   help="Assumed person shoulder width (for alt distance estimate)")
    p.add_argument("--top-down", action="store_true",
                   help="Generate a top-down trajectory plot from all input frames (sorted by filename)")
    args = p.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model (auto-downloads on first run)
    print(f"[yolo] loading {args.model} ...", flush=True)
    model = YOLO(args.model)

    # Collect files
    if in_path.is_file():
        files = [in_path]
    else:
        files = sorted([f for f in in_path.iterdir()
                       if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
    if not files:
        print(f"No images found at {in_path}", file=sys.stderr); sys.exit(1)

    dets_per_frame: List[List[PersonDetection]] = []
    last_size: tuple[int, int] | None = None
    cam: MonoCamera | None = None

    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"  ! could not read {f}"); continue
        h, w = img.shape[:2]
        if (w, h) != last_size:
            cam = MonoCamera((w, h), args.hfov_deg)
            last_size = (w, h)
        dets = detect_persons(model, img, cam,
                              person_height_m=args.person_height,
                              person_width_m=args.person_width,
                              conf_thresh=args.conf)
        dets_per_frame.append(dets)
        annotated = annotate(img, dets, cam)
        out_f = out_dir / f"{f.stem}_annotated.jpg"
        cv2.imwrite(str(out_f), annotated)

        if dets:
            line = f"{f.name}: {len(dets)} person(s) | "
            for i, d in enumerate(dets[:3]):
                line += (f"[{i}] d={d.distance_m:.2f}m bearing={d.bearing_deg:+.1f}° "
                        f"conf={d.confidence:.2f}  ")
            print(line)
        else:
            print(f"{f.name}: no persons detected")

    if args.top_down and len(files) > 1:
        top_down_path = out_dir / "trajectory_top_down.png"
        plot_top_down(dets_per_frame, top_down_path,
                      hfov_deg=args.hfov_deg)
        print(f"[top-down] saved {top_down_path}")

    print(f"\n[done] annotated images in {out_dir}/")


if __name__ == "__main__":
    main()
