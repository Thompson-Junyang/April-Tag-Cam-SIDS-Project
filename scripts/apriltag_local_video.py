# python apriltag_local_video.py   --video "/home/hice1/jtang341/scratch/Apriltag/AprilTag/scripts/WIN_20251102_16_49_00_Pro.mp4"   --workers 24   --ids "22,23"   --tag-size 0.049   --print-every 10




# #!/usr/bin/env python3
# from argparse import ArgumentParser
# import os
# import cv2
# import numpy as np
# import apriltag
# import csv
# import time
# from datetime import datetime

# ################################################################################

# def apriltag_video(input_streams=None,
#                    output_stream=False,          # ★ CHANGED: 默认不导出视频，加速
#                    display_stream=False,         # ★ CHANGED: 默认不显示窗口
#                    detection_window_name='AprilTag',
#                    use_video_time=True):

#     if input_streams is None:
#         input_streams = [
#             "/home/hice1/jtang341/scratch/Apriltag/AprilTag/scripts/WIN_20251102_16_49_00_Pro.mp4"
#         ]

#     parser = ArgumentParser(description='Detect AprilTags from video/camera.')
#     apriltag.add_arguments(parser)
#     options = parser.parse_args()

#     detector = apriltag.Detector(options, searchpath=apriltag._get_dll_path())

#     camera_params = (1408.421651570743, 1405.3445689921414, 1028.1372748266583, 539.4602383823626)
#     tag_size_m = 0.049

#     want_ids = {22, 23}
#     id_a, id_b = sorted(want_ids)

#     for stream in input_streams:
#         if not isinstance(stream, int):
#             if not os.path.exists(str(stream)):
#                 print(f"[ERROR] Video file not found: {stream}")
#                 continue

#         video = cv2.VideoCapture(stream)

#         if isinstance(stream, int):
#             video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
#             video.set(cv2.CAP_PROP_FPS, 30)
#             video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#             video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#             video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

#         if not video.isOpened():
#             print(f"[ERROR] Failed to open stream: {stream}")
#             continue

#         actual_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#         actual_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         actual_fps = video.get(cv2.CAP_PROP_FPS)
#         print(f"[INFO] Capture at {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

#         rows = []   # ★ RESTORED: 只在两个 tag 可见时 append
#         t0_ns = None

#         base_tag = os.path.splitext(os.path.basename(str(stream)))[0] if not isinstance(stream, int) else f"camera_{stream}"
#         csv_name = f"distance_log_{base_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

#         output = None
#         output_path = None

#         frame_i = 0

#         while video.isOpened():
#             success, frame = video.read()
#             if not success:
#                 break

#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#             result, overlay = apriltag.detect_tags(
#                 gray,
#                 detector,
#                 camera_params=camera_params,
#                 tag_size=tag_size_m,
#                 vizualization=0,
#                 verbose=0,
#                 annotation=False
#             )

#             centers = {}
#             tvecs = {}
#             for i in range(0, len(result), 4):
#                 det = result[i]
#                 pose = result[i + 1]
#                 tid = getattr(det, 'tag_id', None)
#                 if tid in (id_a, id_b):
#                     centers[tid] = np.array(det.center, dtype=float)
#                     tvecs[tid] = np.array(pose[:3, 3], dtype=float)

#             # Only log if BOTH tags present ★ RESTORED
#             if id_a in centers and id_b in centers:
#                 pix_dist = float(np.linalg.norm(centers[id_a] - centers[id_b]))
#                 m_dist = None
#                 if id_a in tvecs and id_b in tvecs:
#                     m_dist = float(np.linalg.norm(tvecs[id_a] - tvecs[id_b]))

#                 if use_video_time:
#                     pos_ms = video.get(cv2.CAP_PROP_POS_MSEC)
#                     ts_ms = f"{int(pos_ms):d}ms"
#                     elapsed_ms = int(pos_ms)
#                 else:
#                     now = datetime.now()
#                     ts_ms = f"{now:%Y-%m-%d %H:%M:%S}.{int(now.microsecond/1000):03d}"
#                     now_ns = time.perf_counter_ns()
#                     if t0_ns is None:
#                         t0_ns = now_ns
#                     elapsed_ms = (now_ns - t0_ns) // 1_000_000

#                 rows.append([
#                     ts_ms,
#                     int(elapsed_ms),
#                     f"{pix_dist:.3f}",
#                     "" if m_dist is None else f"{m_dist:.6f}"
#                 ])

#                 text = f"{id_a}↔{id_b}: {pix_dist:.1f}px"
#                 text += f" | {m_dist:.3f} m" if m_dist is not None else ""
#                 text_with_time = f"{text} | {ts_ms}"
#             else:
#                 text_with_time = "tags not both visible"

#             # ★ ADDED: Print every 10 frames
#             if frame_i % 10 == 0:
#                 print(text_with_time)

#             frame_i += 1

#         if rows:
#             with open(csv_name, "w", newline="") as f:
#                 writer = csv.writer(f)
#                 writer.writerow(["timestamp_ms", "elapsed_ms", "pix_dist", "m_dist"])
#                 writer.writerows(rows)
#             print(f"[INFO] Wrote CSV: {os.path.abspath(csv_name)}")

#         if output_stream and output is not None:
#             output.release()
#         video.release()

# ################################################################################

# if __name__ == '__main__':
#     apriltag_video(
#         input_streams=[
#             "/home/hice1/jtang341/scratch/Apriltag/AprilTag/scripts/WIN_20251102_16_49_00_Pro.mp4"
#         ],
#         output_stream=False,
#         display_stream=False,
#         use_video_time=True
#     )




#!/usr/bin/env python3
from argparse import ArgumentParser
import os
import cv2
import numpy as np
import apriltag
import csv
import time
from datetime import datetime
from multiprocessing import Pool, current_process, get_start_method
import math

# -----------------------------
# 工具：读取视频基础信息
# -----------------------------
def _probe_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {path}")
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 可能不精确，但够分段
    cap.release()
    return w, h, fps, total

# -----------------------------
# 单个分段的处理函数（在子进程里运行）
# 输入：视频路径、起止帧下标 [start, end)
# 返回：rows 列表，每行 = [timestamp_ms, elapsed_ms, pix_dist, m_dist]
# -----------------------------
def _process_segment(args):
    (
        video_path, start_f, end_f,
        want_ids, camera_params, tag_size_m,
        apriltag_opts, print_every
    ) = args

    pid = os.getpid()
    worker_name = f"W{pid}"

    # 构建 apriltag Detector（对象不可序列化，必须在子进程里创建）
    parser = ArgumentParser(add_help=False)
    apriltag.add_arguments(parser)
    # 造一个空 options，再把我们想要的参数塞进去
    options = parser.parse_args(args=[])
    # 设置一些常用提速或默认参数，命令行也可覆盖，但这里先用调用方给的 apriltag_opts
    for k, v in apriltag_opts.items():
        setattr(options, k, v)

    detector = apriltag.Detector(options, searchpath=apriltag._get_dll_path())

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    # seek 到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    id_a, id_b = sorted(want_ids)
    rows = []
    local_idx = 0  # 片段内帧计数，用于节流打印

    # 为了“每10帧打印一次”，我们在各 worker 内独立计数
    while True:
        # 当前帧号若越界，退出
        cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if cur >= end_f:
            break

        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 调用与你原代码一致的检测 API（保留相机内参与尺寸）
        # 不做任何 downscale / 改分辨率
        result, _overlay = apriltag.detect_tags(
            gray,
            detector,
            camera_params=camera_params,
            tag_size=tag_size_m,
            vizualization=0,
            verbose=0,
            annotation=False
        )

        centers, tvecs = {}, {}
        for i in range(0, len(result), 4):
            det = result[i]
            pose = result[i + 1]
            tid = getattr(det, 'tag_id', None)
            if tid in (id_a, id_b):
                centers[tid] = np.array(det.center, dtype=float)
                tvecs[tid]   = np.array(pose[:3, 3], dtype=float)

        # 只在两 tag 同时可见时写 CSV（保持你原本行为）
        if id_a in centers and id_b in centers:
            pix_dist = float(np.linalg.norm(centers[id_a] - centers[id_b]))
            m_dist = None
            if id_a in tvecs and id_b in tvecs:
                m_dist = float(np.linalg.norm(tvecs[id_a] - tvecs[id_b]))

            # 使用视频时间戳（毫秒）
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            ts_ms = f"{int(pos_ms):d}ms"
            elapsed_ms = int(pos_ms)

            rows.append([
                ts_ms,
                elapsed_ms,
                f"{pix_dist:.3f}",
                "" if m_dist is None else f"{m_dist:.6f}"
            ])

            # 每10帧打印一次（按worker节流）
            if print_every > 0 and (local_idx % print_every == 0):
                text = f"{id_a}↔{id_b}: {pix_dist:.1f}px"
                if m_dist is not None:
                    text += f" | {m_dist:.3f} m"
                print(f"[{worker_name}] {text} | {ts_ms}")

        local_idx += 1

    cap.release()
    return rows

# -----------------------------
# 主流程：切分任务 + 合并 CSV
# -----------------------------
def run_multicore(video_path,
                  workers,
                  want_ids,
                  camera_params,
                  tag_size_m,
                  apriltag_opts,
                  print_every=10):

    w, h, fps, total = _probe_video(video_path)
    print(f"[INFO] Opened: {video_path}")
    print(f"[INFO] {w}x{h} @ {fps:.2f} FPS, frames={total}")

    # 避免 total 为 0 的极端情况
    if total <= 0:
        # 退化为单进程顺序读取（直到读不到为止）
        total = int(60 * fps)  # 先给一个估算，worker里按read()结束

    # 切分为 workers 份
    workers = max(1, workers)
    seg_size = math.ceil(total / workers)
    tasks = []
    for i in range(workers):
        start_f = i * seg_size
        end_f   = min((i + 1) * seg_size, total)
        if start_f >= end_f:
            continue
        tasks.append((
            video_path, start_f, end_f,
            want_ids, camera_params, tag_size_m,
            apriltag_opts, print_every
        ))

    print(f"[INFO] Dispatch {len(tasks)} segments to {workers} workers")

    # 运行并收集
    with Pool(processes=workers) as pool:
        results = pool.map(_process_segment, tasks)

    # 合并并按 elapsed_ms 排序（视频时间戳）
    merged = []
    for part in results:
        merged.extend(part)

    merged.sort(key=lambda r: r[1])  # 按 elapsed_ms 升序

    # 写 CSV
    base = os.path.splitext(os.path.basename(video_path))[0]
    csv_name = f"distance_log_{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    if merged:
        with open(csv_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_ms", "elapsed_ms", "pix_dist", "m_dist"])
            writer.writerows(merged)
        print(f"[INFO] Wrote CSV: {os.path.abspath(csv_name)}")
    else:
        print("[WARN] No rows to write (no frames with both tags visible).")

# -----------------------------
# 入口
# -----------------------------
def main():
    parser = ArgumentParser(description="Multi-core AprilTag distance from recorded video.")
    parser.add_argument("--video", type=str, required=True, help="Absolute path to video file")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of worker processes")
    parser.add_argument("--ids", type=str, default="22,23", help='Two tag ids, e.g. "22,23"')
    parser.add_argument("--tag-size", type=float, default=0.049, help="Tag black square size in meters")
    parser.add_argument("--print-every", type=int, default=10, help="Print to terminal every K frames (per worker)")

    # 可选：传递一些 apriltag 的性能参数（不改变分辨率）
    parser.add_argument("--nthreads", type=int, default=1, help="apriltag internal threads per worker")
    parser.add_argument("--quad-decimate", type=float, default=1.0, help="apriltag quad decimate (>=1.0, bigger = faster)")
    parser.add_argument("--refine-edges", type=int, default=1, help="apriltag refine edges (1=yes, 0=no)")
    parser.add_argument("--families", type=str, default="tag36h11", help="tag family")

    args = parser.parse_args()

    # 解析 ids
    try:
        want_ids = set(int(x.strip()) for x in args.ids.split(",")[:2])
        if len(want_ids) != 2:
            raise ValueError
    except Exception:
        raise ValueError("--ids must contain two integers, e.g., --ids '22,23'")

    # 你的相机内参（不缩放）
    camera_params = (1408.421651570743, 1405.3445689921414, 1028.1372748266583, 539.4602383823626)

    # 聚合 apriltag 选项（传入子进程）
    apriltag_opts = {
        "nthreads": max(1, int(args.nthreads)),
        "quad_decimate": float(args.quad_decimate),
        "refine_edges": int(args.refine_edges),
        "families": args.families,
    }

    # 运行
    run_multicore(
        video_path=os.path.abspath(args.video),
        workers=int(args.workers),
        want_ids=want_ids,
        camera_params=camera_params,
        tag_size_m=float(args.tag_size),
        apriltag_opts=apriltag_opts,
        print_every=int(args.print_every),
    )

if __name__ == "__main__":
    # 在 Linux 下默认 spawn/fork 都可，这里不强制修改
    main()
