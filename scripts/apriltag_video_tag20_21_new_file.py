# #!/usr/bin/env python

# from argparse import ArgumentParser
# import os
# import cv2
# import numpy as np
# import apriltag
# import csv                     # ★ ADDED
# import time                    # ★ ADDED
# from datetime import datetime  # ★ ADDED

# ################################################################################

# def apriltag_video(
#     input_streams=[0],
#     output_stream=False,
#     display_stream=True,
#     detection_window_name='AprilTag'
# ):
#     """
#     Detect ONLY the two IDs in want_ids from camera or video.
#     When both are visible, draw a line and compute:
#       - pixel distance
#       - 3D distance (meters)
#     Also logs CSV with video-time timestamps:
#       timestamp_ms (mm:ss.mmm), elapsed_ms (int), pix_dist, m_dist
#     """

#     parser = ArgumentParser(description='Detect AprilTags from camera/video.')
#     apriltag.add_arguments(parser)  # keep apriltag’s own options

#     # ---------------- our minimal args (不大改结构) ----------------
#     parser.add_argument('--camera', type=int, default=None, help='Camera index, e.g., 0')
#     parser.add_argument('--video', type=str, default=None, help='Path to a local video file')
#     parser.add_argument('--ids', type=str, default='22,23', help='Two tag ids like "22,23"')  # ★ ADDED
#     parser.add_argument('--tag-size', type=float, default=0.049, help='Tag black square size in meters')  # ★ ADDED
#     parser.add_argument('--no-display', action='store_true', help='Disable window display')  # ★ ADDED
#     parser.add_argument('--output', action='store_true', help='Save annotated video to AVI') # ★ ADDED
#     options = parser.parse_args()

#     # ---- IDs you want to track (默认 22,23) ----
#     try:
#         want_ids = set(int(x.strip()) for x in options.ids.split(',')[:2])  # ★ ADDED
#     except Exception:
#         want_ids = {22, 23}  # fallback
#     id_a, id_b = sorted(list(want_ids))  # ★ ADDED

#     # AprilTag detector via apriltag options (保持原生参数风格)
#     detector = apriltag.Detector(options, searchpath=apriltag._get_dll_path())

#     # ---- 你的相机内参（在标定分辨率下）----
#     camera_params = (1408.421651570743, 1405.3445689921414, 1028.1372748266583, 539.4602383823626)
#     # fx, fy, cx, cy
#     tag_size_m = options.tag_size  # ★ CHANGED: 改为可通过参数指定

#     # ---- 输入来源选择（摄像头或本地视频）----  # ★ ADDED
#     streams = []
#     if options.video is not None:
#         streams = [options.video]
#     elif options.camera is not None:
#         streams = [int(options.camera)]
#     else:
#         streams = input_streams  # 兼容原有函数签名

#     # ---- 显示/输出控制 ----  # ★ ADDED
#     display_stream = False if options.no_display else display_stream
#     output_stream = True if options.output else output_stream

#     for stream in streams:
#         video = cv2.VideoCapture(stream)

#         # 若是摄像头（整数索引），请求 1080p + MJPG，平滑低延迟
#         if isinstance(stream, int):                # ★ ADDED
#             video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
#             video.set(cv2.CAP_PROP_FPS, 30)
#             video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#             video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#             video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

#         actual_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # ★ ADDED
#         actual_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # ★ ADDED
#         fps = video.get(cv2.CAP_PROP_FPS) or 30.0            # ★ CHANGED: 后面用它算“视频时间戳”
#         print(f"[INFO] Capture at {actual_w}x{actual_h} @ {fps:.1f} FPS")  # 信息行，可保留

#         # CSV 累积与命名（按输入源区分文件名）  # ★ ADDED
#         rows = []  # [timestamp_ms(mm:ss.mmm), elapsed_ms, pix_dist, m_dist]
#         if isinstance(stream, int):
#             csv_name = f"distance_log_camera{stream}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
#         else:
#             base = os.path.splitext(os.path.basename(str(stream)))[0]
#             csv_name = f"distance_log_{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

#         # 可选：输出编码视频（叠加文字）
#         output = None
#         if output_stream:
#             codec = cv2.VideoWriter_fourcc(*'XVID')
#             out_path = (
#                 f"camera_{stream}.avi" if isinstance(stream, int)
#                 else f"{os.path.splitext(os.path.basename(str(stream)))[0]}.avi"
#             )
#             out_dir = os.path.join('..', 'media', 'output')
#             os.makedirs(out_dir, exist_ok=True)
#             output_path = os.path.join(out_dir, out_path)
#             output = cv2.VideoWriter(output_path, codec, fps, (actual_w, actual_h))

#         frame_i = 0  # 用来计算“视频时间”

#         while video.isOpened():
#             success, frame = video.read()
#             if not success:
#                 break

#             # 运行检测（不画 apriltag 内置标注）
#             result, overlay = apriltag.detect_tags(
#                 frame,
#                 detector,
#                 camera_params=camera_params,
#                 tag_size=tag_size_m,
#                 vizualization=0,
#                 verbose=0,
#                 annotation=False
#             )

#             # 收集我们关注的两个 ID 的中心与位姿
#             centers = {}
#             tvecs = {}
#             for i in range(0, len(result), 4):
#                 det = result[i]        # apriltag.Detection
#                 pose = result[i + 1]   # 3x4 pose (tag->camera)
#                 tid = getattr(det, 'tag_id', None)
#                 if tid in want_ids:
#                     centers[tid] = np.array(det.center, dtype=float)
#                     tvecs[tid] = np.array(pose[:3, 3], dtype=float)

#             # 两个标签都在画面里才计算距离并记录
#             if id_a in centers and id_b in centers:
#                 pix_dist = float(np.linalg.norm(centers[id_a] - centers[id_b]))

#                 m_dist = None
#                 if id_a in tvecs and id_b in tvecs:
#                     m_dist = float(np.linalg.norm(tvecs[id_a] - tvecs[id_b]))

#                 # 画线+叠加信息（显示时可见）
#                 p0 = tuple(np.round(centers[id_a]).astype(int))
#                 p1 = tuple(np.round(centers[id_b]).astype(int))
#                 cv2.line(overlay, p0, p1, (0, 255, 255), 2)

#                 # ★★ 关键改动：使用“视频时间戳”，而不是真实世界时间 ★★
#                 video_time_s = frame_i / float(fps)                       # ★ ADDED
#                 elapsed_ms = int(video_time_s * 1000)                     # ★ ADDED
#                 mm = int(video_time_s // 60)                              # ★ ADDED
#                 ss = int(video_time_s % 60)                               # ★ ADDED
#                 ms = int((video_time_s * 1000) % 1000)                    # ★ ADDED
#                 ts_ms = f"{mm:02d}:{ss:02d}.{ms:03d}"                     # ★ ADDED

#                 # 叠加显示文本（不在终端打印）        # ★ CHANGED: 不再 print 每帧
#                 if m_dist is not None:
#                     text = f"{id_a}↔{id_b}: {pix_dist:.1f}px | {m_dist:.3f} m | {ts_ms}"
#                 else:
#                     text = f"{id_a}↔{id_b}: {pix_dist:.1f}px | {ts_ms}"

#                 mid = (int((p0[0] + p1[0]) / 2), int((p0[1] + p1[1]) / 2) - 10)
#                 cv2.putText(overlay, text, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

#                 # ★★ 关键改动：CSV 写入“每一帧”（不节流） ★★
#                 rows.append([
#                     ts_ms,
#                     int(elapsed_ms),
#                     f"{pix_dist:.3f}",
#                     "" if m_dist is None else f"{m_dist:.6f}"
#                 ])

#             frame_i += 1  # ★ ADDED

#             # 可选：保存叠加视频
#             if output_stream and output is not None:
#                 output.write(overlay)

#             # 可选：显示窗口（按空格退出）
#             if display_stream:
#                 cv2.imshow(detection_window_name, overlay)
#                 if cv2.waitKey(1) & 0xFF == ord(' '):
#                     break

#         # 写出 CSV（如果收集到数据）
#         if rows:  # ★ ADDED
#             with open(csv_name, "w", newline="", encoding="utf-8") as f:
#                 writer = csv.writer(f)
#                 writer.writerow(["timestamp_ms", "elapsed_ms", "pix_dist", "m_dist"])
#                 writer.writerows(rows)
#             print(f"[INFO] Wrote CSV: {os.path.abspath(csv_name)}")

#         if output_stream and output is not None:
#             output.release()
#         video.release()

# ################################################################################

# if __name__ == '__main__':
#     apriltag_video()







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
#                    output_stream=True,
#                    display_stream=True,
#                    detection_window_name='AprilTag',
#                    use_video_time=True):
#     """
#     Detect ONLY the two IDs in want_ids from video/camera. If both are visible,
#     draw a line between their image centers and display:
#       - pixel distance between centers
#       - 3D distance between tag centers (in meters), using pose estimation
#       - timestamp (either wall-clock or video time)
#     Logs CSV rows: timestamp_ms, elapsed_ms, pix_dist, m_dist
#     """

#     # 默认本地视频（可在 __main__ 里覆盖）
#     if input_streams is None:
#         input_streams = [
#             "/home/t/Proj/Apriltag/April-Tag-Cam-SIDS-Project/media/WIN_20251102_16_49_00_Pro.mp4"
#         ]

#     parser = ArgumentParser(description='Detect AprilTags from video/camera.')
#     # 保留 apriltag 的原生参数（如 nthreads, quad_decimate, refine_edges 等）
#     apriltag.add_arguments(parser)
#     options = parser.parse_args()

#     # AprilTag detector
#     detector = apriltag.Detector(options, searchpath=apriltag._get_dll_path())

#     # 相机内参（按你的标定分辨率）
#     # fx, fy, cx, cy
#     camera_params = (1408.421651570743, 1405.3445689921414, 1028.1372748266583, 539.4602383823626)
#     tag_size_m = 0.049  # 单位：米

#     # 只跟踪这两个 ID
#     want_ids = {22, 23}
#     id_a, id_b = sorted(want_ids)

#     for stream in input_streams:
#         # ---------- 打开前的防护检查 ----------
#         if not isinstance(stream, int):
#             # 文件路径检查
#             if not os.path.exists(str(stream)):
#                 print(f"[ERROR] Video file not found: {stream}")
#                 continue

#         video = cv2.VideoCapture(stream)

#         # 若是摄像头（整数索引），请求 1080p + MJPG，低延迟
#         if isinstance(stream, int):
#             video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
#             video.set(cv2.CAP_PROP_FPS, 30)
#             video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#             video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#             video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

#         # 打开失败直接跳过，避免 0x0 尺寸触发后续崩溃
#         if not video.isOpened():
#             print(f"[ERROR] Failed to open stream: {stream}")
#             continue

#         actual_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#         actual_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         actual_fps = video.get(cv2.CAP_PROP_FPS)
#         print(f"[INFO] Capture at {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

#         # CSV 累积
#         rows = []  # [timestamp_ms, elapsed_ms, pix_dist, m_dist]
#         t0_ns = None  # 仅在 use_video_time=False 时使用

#         # CSV 文件名包含源标识
#         base_tag = os.path.splitext(os.path.basename(str(stream)))[0] if not isinstance(stream, int) else f"camera_{stream}"
#         csv_name = f"distance_log_{base_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

#         # 输出视频延后到首帧成功读取后再初始化
#         output = None
#         output_path = None

#         frame_i = 0

#         while video.isOpened():
#             success, frame = video.read()
#             if not success:
#                 break

#             # 首帧到手时，若需要，初始化 VideoWriter（用实际帧尺寸与兜底 FPS）
#             if output_stream and output is None:
#                 h, w = frame.shape[:2]
#                 fps_v = video.get(cv2.CAP_PROP_FPS)
#                 fps_v = int(fps_v) if fps_v and fps_v > 0 else 30
#                 codec = cv2.VideoWriter_fourcc(*'XVID')
#                 if isinstance(stream, int):
#                     output_path = os.path.join('..', 'media', 'output', f'camera_{stream}.avi')
#                 else:
#                     base = os.path.splitext(os.path.basename(str(stream)))[0] + '.avi'
#                     output_path = os.path.join('..', 'media', 'output', base)
#                 os.makedirs(os.path.dirname(output_path), exist_ok=True)
#                 output = cv2.VideoWriter(output_path, codec, fps_v, (w, h))
#                 print(f"[INFO] Writing annotated video to: {os.path.abspath(output_path)}")

#             # 运行检测（不使用 apriltag 内置叠加）
#             result, overlay = apriltag.detect_tags(
#                 frame,
#                 detector,
#                 camera_params=camera_params,
#                 tag_size=tag_size_m,
#                 vizualization=0,
#                 verbose=0,
#                 annotation=False
#             )

#             # 提取我们关注的两个 ID 的中心与位姿
#             centers = {}
#             tvecs = {}
#             for i in range(0, len(result), 4):
#                 det = result[i]      # apriltag.Detection
#                 pose = result[i + 1] # 3x4 pose matrix (tag->camera)
#                 tid = getattr(det, 'tag_id', None)
#                 if tid in (id_a, id_b):
#                     centers[tid] = np.array(det.center, dtype=float)
#                     tvecs[tid] = np.array(pose[:3, 3], dtype=float)

#             # 两个都在画面里才计算距离与记录
#             if id_a in centers and id_b in centers:
#                 # 2D 像素距离
#                 pix_dist = float(np.linalg.norm(centers[id_a] - centers[id_b]))

#                 # 3D 米距离（基于相机坐标系中的两标签平移向量）
#                 m_dist = None
#                 if id_a in tvecs and id_b in tvecs:
#                     m_dist = float(np.linalg.norm(tvecs[id_a] - tvecs[id_b]))

#                 # 画线与文字
#                 p0 = tuple(np.round(centers[id_a]).astype(int))
#                 p1 = tuple(np.round(centers[id_b]).astype(int))
#                 cv2.line(overlay, p0, p1, (0, 255, 255), 2)

#                 # 时间戳
#                 if use_video_time:
#                     # 使用“视频内时间戳”
#                     pos_ms = video.get(cv2.CAP_PROP_POS_MSEC)  # 从视频起点的毫秒
#                     ts_ms = f"{int(pos_ms):d}ms"
#                     elapsed_ms = int(pos_ms)
#                 else:
#                     # 使用墙钟时间 + 高精度相对时间
#                     now = datetime.now()
#                     ts_ms = f"{now:%Y-%m-%d %H:%M:%S}.{int(now.microsecond/1000):03d}"
#                     now_ns = time.perf_counter_ns()
#                     if t0_ns is None:
#                         t0_ns = now_ns
#                     elapsed_ms = (now_ns - t0_ns) // 1_000_000

#                 # 标签文本
#                 if m_dist is not None:
#                     text = f"{id_a}↔{id_b}: {pix_dist:.1f}px | {m_dist:.3f} m"
#                 else:
#                     text = f"{id_a}↔{id_b}: {pix_dist:.1f}px"
#                 text_with_time = f"{text} | {ts_ms}"

#                 mid = (int((p0[0] + p1[0]) / 2), int((p0[1] + p1[1]) / 2) - 10)
#                 cv2.putText(overlay, text_with_time, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

#                 # 节流打印
#                 if frame_i % 10 == 0:
#                     print(text_with_time)

#                 # CSV 记录
#                 rows.append([
#                     ts_ms,
#                     int(elapsed_ms),
#                     f"{pix_dist:.3f}",
#                     "" if m_dist is None else f"{m_dist:.6f}"
#                 ])

#             frame_i += 1

#             # 写输出视频帧
#             if output_stream and output is not None:
#                 output.write(overlay)

#             # 可视化
#             if display_stream:
#                 cv2.imshow(detection_window_name, overlay)
#                 if cv2.waitKey(1) & 0xFF == ord(' '):
#                     break

#         # 写 CSV
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
#     # 把路径换成你的真实视频路径；要用摄像头就传 [0]
#     apriltag_video(
#         input_streams=[
#             "/home/t/Proj/Apriltag/April-Tag-Cam-SIDS-Project/media/WIN_20251102_16_49_00_Pro.mp4"
#         ],
#         output_stream=True,   # 需要导出叠加视频就 True
#         display_stream=True,  # 无显示环境(例如远程/WSL)改 False
#         use_video_time=True   # True=视频时间戳；False=墙钟时间
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

################################################################################

def apriltag_video(input_streams=None,
                   output_stream=False,          # ★ CHANGED: 默认不导出视频，加速
                   display_stream=False,         # ★ CHANGED: 默认不显示窗口
                   detection_window_name='AprilTag',
                   use_video_time=True):

    if input_streams is None:
        input_streams = [
            "/home/t/Proj/Apriltag/April-Tag-Cam-SIDS-Project/media/WIN_20251102_16_49_00_Pro.mp4"
        ]

    parser = ArgumentParser(description='Detect AprilTags from video/camera.')
    apriltag.add_arguments(parser)
    options = parser.parse_args()

    detector = apriltag.Detector(options, searchpath=apriltag._get_dll_path())

    camera_params = (1408.421651570743, 1405.3445689921414, 1028.1372748266583, 539.4602383823626)
    tag_size_m = 0.049

    want_ids = {22, 23}
    id_a, id_b = sorted(want_ids)

    for stream in input_streams:
        if not isinstance(stream, int):
            if not os.path.exists(str(stream)):
                print(f"[ERROR] Video file not found: {stream}")
                continue

        video = cv2.VideoCapture(stream)

        if isinstance(stream, int):
            video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            video.set(cv2.CAP_PROP_FPS, 30)
            video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not video.isOpened():
            print(f"[ERROR] Failed to open stream: {stream}")
            continue

        actual_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = video.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] Capture at {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

        rows = []   # ★ RESTORED: 只在两个 tag 可见时 append
        t0_ns = None

        base_tag = os.path.splitext(os.path.basename(str(stream)))[0] if not isinstance(stream, int) else f"camera_{stream}"
        csv_name = f"distance_log_{base_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        output = None
        output_path = None

        frame_i = 0

        while video.isOpened():
            success, frame = video.read()
            if not success:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            result, overlay = apriltag.detect_tags(
                gray,
                detector,
                camera_params=camera_params,
                tag_size=tag_size_m,
                vizualization=0,
                verbose=0,
                annotation=False
            )

            centers = {}
            tvecs = {}
            for i in range(0, len(result), 4):
                det = result[i]
                pose = result[i + 1]
                tid = getattr(det, 'tag_id', None)
                if tid in (id_a, id_b):
                    centers[tid] = np.array(det.center, dtype=float)
                    tvecs[tid] = np.array(pose[:3, 3], dtype=float)

            # Only log if BOTH tags present ★ RESTORED
            if id_a in centers and id_b in centers:
                pix_dist = float(np.linalg.norm(centers[id_a] - centers[id_b]))
                m_dist = None
                if id_a in tvecs and id_b in tvecs:
                    m_dist = float(np.linalg.norm(tvecs[id_a] - tvecs[id_b]))

                if use_video_time:
                    pos_ms = video.get(cv2.CAP_PROP_POS_MSEC)
                    ts_ms = f"{int(pos_ms):d}ms"
                    elapsed_ms = int(pos_ms)
                else:
                    now = datetime.now()
                    ts_ms = f"{now:%Y-%m-%d %H:%M:%S}.{int(now.microsecond/1000):03d}"
                    now_ns = time.perf_counter_ns()
                    if t0_ns is None:
                        t0_ns = now_ns
                    elapsed_ms = (now_ns - t0_ns) // 1_000_000

                rows.append([
                    ts_ms,
                    int(elapsed_ms),
                    f"{pix_dist:.3f}",
                    "" if m_dist is None else f"{m_dist:.6f}"
                ])

                text = f"{id_a}↔{id_b}: {pix_dist:.1f}px"
                text += f" | {m_dist:.3f} m" if m_dist is not None else ""
                text_with_time = f"{text} | {ts_ms}"
            else:
                text_with_time = "tags not both visible"

            # ★ ADDED: Print every 10 frames
            if frame_i % 10 == 0:
                print(text_with_time)

            frame_i += 1

        if rows:
            with open(csv_name, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp_ms", "elapsed_ms", "pix_dist", "m_dist"])
                writer.writerows(rows)
            print(f"[INFO] Wrote CSV: {os.path.abspath(csv_name)}")

        if output_stream and output is not None:
            output.release()
        video.release()

################################################################################

if __name__ == '__main__':
    apriltag_video(
        input_streams=[
            "/home/t/Proj/Apriltag/April-Tag-Cam-SIDS-Project/media/WIN_20251102_16_49_00_Pro.mp4"
        ],
        output_stream=False,
        display_stream=False,
        use_video_time=True
    )
