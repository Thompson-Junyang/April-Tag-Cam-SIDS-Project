# #!/usr/bin/env python

# from argparse import ArgumentParser
# import os
# import cv2
# import numpy as np
# import apriltag

# ################################################################################

# def apriltag_video(input_streams=[0],
#                    output_stream=False,
#                    display_stream=True,
#                    detection_window_name='AprilTag'):
#     """
#     Detect ONLY tags 20 and 21 from video. If both are visible,
#     draw a line between their image centers and display:
#       - pixel distance between centers
#       - 3D distance between tag centers (in meters), using pose estimation
#     """

#     parser = ArgumentParser(description='Detect AprilTags from video stream.')
#     apriltag.add_arguments(parser)
#     options = parser.parse_args()

#     # -------- AprilTag detector (accuracy over speed) --------
#     # NOTE: Your apriltag version doesn't support quad_sigma; omit it.
#     detector = apriltag.Detector(
#         apriltag.DetectorOptions(
#             families='tag36h11',
#             quad_decimate=1.0,
#             refine_edges=True
#         ),
#         searchpath=apriltag._get_dll_path()
#     )

#     # -------- Your calibrated camera intrinsics (AT CALIBRATION RESOLUTION) --------
#     fx_cal, fy_cal, cx_cal, cy_cal = (
#         1456.6855555383966,  # fx
#         1452.2059008627816,  # fy
#         966.8814607344126,   # cx
#         549.333443447071     # cy
#     )

#     # Your distortion coefficients (k1, k2, p1, p2, k3)
#     distCoeffs = np.array([
#         0.06896850181732163,
#         0.4101164004003805,
#         -0.0029444965742958665,
#         0.002356027867842563,
#         -2.0391513648270556
#     ], dtype=np.float32)

#     # If you KNOW your calibration image size, set explicitly:
#     CALIB_W, CALIB_H = 1920, 1080

#     # -------- Tag setup --------
#     tag_size_m = 0.049  # meters; OUTER black square side length (measure it)
#     want_ids = {20, 21}

#     # -------- Helpers --------
#     def corners_inside_image(corners_xy, H, W, win_half=2):
#         """
#         Check all corners lie within image bounds, leaving a small margin equal to window half-size
#         to keep cornerSubPix's window fully inside.
#         corners_xy: (4,2) float
#         """
#         x_ok = (corners_xy[:, 0] >= win_half) & (corners_xy[:, 0] < (W - win_half))
#         y_ok = (corners_xy[:, 1] >= win_half) & (corners_xy[:, 1] < (H - win_half))
#         return bool(np.all(x_ok & y_ok))

#     def safe_corner_subpix(gray, corners_xy, win=(5, 5)):
#         """
#         Try sub-pixel refinement if corners are inside image with margin.
#         Return possibly-refined corners (4,2). If not safe, return original.
#         """
#         H, W = gray.shape[:2]
#         win_half = max(win[0] // 2, win[1] // 2)
#         if not corners_inside_image(corners_xy, H, W, win_half=win_half):
#             return corners_xy  # skip refinement near edges

#         pts = corners_xy.astype(np.float32).reshape(-1, 1, 2)
#         cv2.cornerSubPix(
#             gray, pts, win, (-1, -1),
#             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-2)
#         )
#         return pts.reshape(-1, 2)

#     for stream in input_streams:
#         video = cv2.VideoCapture(stream)

#         # Grab one frame to get actual capture size
#         ok_first, frame0 = video.read()
#         if not ok_first:
#             print("Failed to read from video source.")
#             return

#         H, W = frame0.shape[:2]

#         # ---- Scale intrinsics to the live (W,H) ----
#         sx, sy = W / float(CALIB_W), H / float(CALIB_H)
#         fx, fy = fx_cal * sx, fy_cal * sy
#         cx, cy = cx_cal * sx, cy_cal * sy
#         K = np.array([[fx, 0.0, cx],
#                       [0.0, fy, cy],
#                       [0.0, 0.0, 1.0]], dtype=np.float32)

#         # ---- Undistort to rectified image with newK, use maps for speed ----
#         newK, _ = cv2.getOptimalNewCameraMatrix(K, distCoeffs, (W, H), 0)
#         map1, map2 = cv2.initUndistortRectifyMap(K, distCoeffs, None, newK, (W, H), cv2.CV_32FC1)
#         zeroDist = np.zeros((5, 1), dtype=np.float32)  # for PnP on undistorted image

#         # Prepare output writer if requested
#         output = None
#         if output_stream:
#             fps = int(video.get(cv2.CAP_PROP_FPS)) or 30
#             codec = cv2.VideoWriter_fourcc(*'XVID')
#             if isinstance(stream, int):
#                 output_path = os.path.join('..', 'media', 'output', f'camera_{stream}.avi')
#             else:
#                 base = os.path.splitext(os.path.basename(str(stream)))[0] + '.avi'
#                 output_path = os.path.join('..', 'media', 'output', base)
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)
#             output = cv2.VideoWriter(output_path, codec, fps, (W, H))

#         # -------- 3D model corners in tag frame (meters), order MUST MATCH det.corners (TL, TR, BR, BL) --------
#         s = tag_size_m / 2.0
#         OBJ_TL = (-s, +s, 0.0)
#         OBJ_TR = (+s, +s, 0.0)
#         OBJ_BR = (+s, -s, 0.0)
#         OBJ_BL = (-s, -s, 0.0)
#         obj_corners = np.array([OBJ_TL, OBJ_TR, OBJ_BR, OBJ_BL], dtype=np.float32)

#         # For temporal consistency (optional penalty in selection)
#         prev_rvec, prev_tvec = {}, {}

#         def solve_pose_ippe_stable(det, gray_undist, tid):
#             """
#             Solve pose for a single tag on the UNDISTORTED image.
#             Uses IPPE_SQUARE and selects the solution with the lowest reprojection error,
#             with a small temporal penalty to avoid solution flipping.
#             """
#             # Detected corners (TL, TR, BR, BL). Shape (4,2).
#             img_pts = det.corners.astype(np.float32)

#             # Safe sub-pixel refinement (skip near edges)
#             img_pts = safe_corner_subpix(gray_undist, img_pts, win=(5, 5))

#             # Solve with IPPE_SQUARE (two possible solutions)
#             ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
#                 objectPoints=obj_corners,
#                 imagePoints=img_pts.reshape(-1, 1, 2),
#                 cameraMatrix=newK,
#                 distCoeffs=zeroDist,
#                 flags=cv2.SOLVEPNP_IPPE_SQUARE
#             )
#             if not ok or len(rvecs) == 0:
#                 return False, None

#             # Pick best by reprojection error (+ small temporal penalty)
#             best, best_err = None, 1e9
#             for rvec, tvec in zip(rvecs, tvecs):
#                 proj, _ = cv2.projectPoints(obj_corners, rvec, tvec, newK, zeroDist)
#                 err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1)))

#                 if tid in prev_rvec and tid in prev_tvec:
#                     err += 0.05 * np.linalg.norm(rvec.reshape(-1) - prev_rvec[tid])
#                     err += 0.01 * np.linalg.norm(tvec.reshape(-1) - prev_tvec[tid])

#                 if err < best_err:
#                     best_err = err
#                     best = (rvec.reshape(-1), tvec.reshape(-1))

#             if best is None:
#                 return False, None

#             rvec_sel, tvec_sel = best
#             prev_rvec[tid], prev_tvec[tid] = rvec_sel, tvec_sel
#             return True, tvec_sel

#         # Process the already-captured first frame, then the rest
#         first = True
#         while True:
#             if first:
#                 frame = frame0
#                 first = False
#             else:
#                 ok, frame = video.read()
#                 if not ok:
#                     break

#             # Undistort / rectify to match newK, zeroDist
#             frame_undist = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
#             gray_undist = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)

#             # Detect on UNDISTORTED frame; pass newK (coherent with the image)
#             result, overlay = apriltag.detect_tags(
#                 frame_undist,
#                 detector,
#                 camera_params=(float(newK[0, 0]), float(newK[1, 1]), float(newK[0, 2]), float(newK[1, 2])),
#                 tag_size=tag_size_m,
#                 vizualization=0,
#                 verbose=0,
#                 annotation=False
#             )

#             centers = {}
#             tvecs = {}

#             for i in range(0, len(result), 4):
#                 det = result[i]        # apriltag.Detection
#                 tid = getattr(det, 'tag_id', None)
#                 if tid in want_ids:
#                     centers[tid] = np.array(det.center, dtype=float)
#                     ok_pose, tvec = solve_pose_ippe_stable(det, gray_undist, tid)
#                     if ok_pose:
#                         tvecs[tid] = tvec

#             # If both present, compute distances and draw
#             if 20 in centers and 21 in centers:
#                 # Pixel distance (UNDISTORTED image)
#                 pix_dist = float(np.linalg.norm(centers[20] - centers[21]))

#                 # 3D distance in meters (camera frame)
#                 m_dist = None
#                 if 20 in tvecs and 21 in tvecs:
#                     m_dist = float(np.linalg.norm(tvecs[20] - tvecs[21]))

#                 # Draw a line and text
#                 p0 = tuple(np.round(centers[20]).astype(int))
#                 p1 = tuple(np.round(centers[21]).astype(int))
#                 cv2.line(overlay, p0, p1, (0, 255, 255), 2)

#                 if m_dist is not None:
#                     text = f"20↔21: {pix_dist:.1f}px | {m_dist:.3f} m"
#                 else:
#                     text = f"20↔21: {pix_dist:.1f}px"

#                 mid = (int((p0[0] + p1[0]) / 2), int((p0[1] + p1[1]) / 2) - 10)
#                 cv2.putText(overlay, text, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
#                 print(text)

#             # Output/display
#             if output_stream and output is not None:
#                 output.write(overlay)

#             if display_stream:
#                 cv2.imshow(detection_window_name, overlay)
#                 if cv2.waitKey(1) & 0xFF == ord(' '):
#                     break

#         if output_stream and output is not None:
#             output.release()
#         video.release()

# ################################################################################

# if __name__ == '__main__':
#     apriltag_video()































# #!/usr/bin/env python

# from argparse import ArgumentParser
# import os
# import cv2
# import numpy as np
# import apriltag

# ################################################################################

# def apriltag_video(input_streams=[0],
#                    output_stream=False,
#                    display_stream=True,
#                    detection_window_name='AprilTag'):
#     """
#     Detect ONLY tags 20 and 21 from video. If both are visible,
#     draw a line between their image centers and display:
#       - pixel distance between centers
#       - 3D distance between tag centers (in meters), using pose estimation
#     """

#     parser = ArgumentParser(description='Detect AprilTags from video stream.')
#     apriltag.add_arguments(parser)
#     options = parser.parse_args()

#     # Set up a reasonable search path for the apriltag DLL/so/dylib.
#     detector = apriltag.Detector(options, searchpath=apriltag._get_dll_path())

#     # Hard-coded camera intrinsics and tag size (in meters)
#     camera_params = (1408.421651570743, 1405.3445689921414, 1028.1372748266583, 539.4602383823626)
#     # fx, fy, cx, cy
#     tag_size_m = 0.049  # meters

#     want_ids = {20, 21}

#     for stream in input_streams:
#         video = cv2.VideoCapture(stream)

#         # ✅ Force 1080p resolution
#         video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#         video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

#         output = None
#         if output_stream:
#             width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#             height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             fps = int(video.get(cv2.CAP_PROP_FPS)) or 30
#             codec = cv2.VideoWriter_fourcc(*'XVID')
#             if type(stream) != int:
#                 output_path = os.path.join('..', 'media', 'output', os.path.splitext(os.path.basename(str(stream)))[0] + '.avi')
#             else:
#                 output_path = os.path.join('..', 'media', 'output', f'camera_{stream}.avi')
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)
#             output = cv2.VideoWriter(output_path, codec, fps, (width, height))

#         while video.isOpened():
#             success, frame = video.read()
#             if not success:
#                 break

#             # Run detection (no verbose prints, no built-in annotations)
#             result, overlay = apriltag.detect_tags(
#                 frame,
#                 detector,
#                 camera_params=camera_params,
#                 tag_size=tag_size_m,
#                 vizualization=0,
#                 verbose=0,
#                 annotation=False
#             )

#             # Collect centers and poses for tag 20 and 21, if found
#             centers = {}
#             tvecs = {}
#             for i in range(0, len(result), 4):
#                 det = result[i]        # apriltag.Detection
#                 pose = result[i + 1]   # 3x4 pose matrix (tag->camera)
#                 tid = getattr(det, 'tag_id', None)
#                 if tid in want_ids:
#                     centers[tid] = np.array(det.center, dtype=float)
#                     tvecs[tid] = np.array(pose[:3, 3], dtype=float)

#             # If both present, compute distances and draw
#             if 20 in centers and 21 in centers:
#                 # 2D pixel distance
#                 pix_dist = float(np.linalg.norm(centers[20] - centers[21]))

#                 # 3D distance in meters (using tag poses in camera frame)
#                 if 20 in tvecs and 21 in tvecs:
#                     m_dist = float(np.linalg.norm(tvecs[20] - tvecs[21]))
#                 else:
#                     m_dist = None

#                 # Draw a line between the tag centers
#                 p0 = tuple(np.round(centers[20]).astype(int))
#                 p1 = tuple(np.round(centers[21]).astype(int))
#                 cv2.line(overlay, p0, p1, (0, 255, 255), 2)

#                 # Prepare the text
#                 if m_dist is not None:
#                     text = f"20↔21: {pix_dist:.1f}px | {m_dist:.3f} m"
#                 else:
#                     text = f"20↔21: {pix_dist:.1f}px"

#                 # Put the text near the midpoint of the segment
#                 mid = (int((p0[0] + p1[0]) / 2), int((p0[1] + p1[1]) / 2) - 10)
#                 cv2.putText(overlay, text, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

#                 # Also print to terminal
#                 print(text)

#             # Optionally write/display
#             if output_stream and output is not None:
#                 output.write(overlay)

#             if display_stream:
#                 cv2.imshow(detection_window_name, overlay)
#                 # Press space bar to terminate
#                 if cv2.waitKey(1) & 0xFF == ord(' '):
#                     break

#         if output_stream and output is not None:
#             output.release()
#         video.release()

# ################################################################################

# if __name__ == '__main__':
#     apriltag_video()




















# #!/usr/bin/env python

# from argparse import ArgumentParser
# import os
# import cv2
# import numpy as np
# import apriltag

# ################################################################################

# def apriltag_video(input_streams=[0],
#                    output_stream=False,
#                    display_stream=True,
#                    detection_window_name='AprilTag'):
#     """
#     Detect ONLY tags 20 and 21 from video. If both are visible,
#     draw a line between their image centers and display:
#       - pixel distance between centers
#       - 3D distance between tag centers (in meters), using pose estimation
#     """

#     parser = ArgumentParser(description='Detect AprilTags from video stream.')
#     apriltag.add_arguments(parser)
#     options = parser.parse_args()

#     # Set up a reasonable search path for the apriltag DLL/so/dylib.
#     detector = apriltag.Detector(options, searchpath=apriltag._get_dll_path())

#     # Hard-coded camera intrinsics and tag size (in meters)
#     camera_params = (1408.421651570743, 1405.3445689921414, 1028.1372748266583, 539.4602383823626)
#     # fx, fy, cx, cy
#     tag_size_m = 0.049  # meters

#     want_ids = {20, 21}

#     for stream in input_streams:
#         video = cv2.VideoCapture(stream)

#         # ---------- ADDED: request smooth 1080p ----------
#         # Use MJPG so the camera sends compressed frames (big perf win on many USB cams)
#         video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
#         video.set(cv2.CAP_PROP_FPS, 30)        # try 15 if bandwidth/CPU is tight
#         video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#         video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#         # Keep the driver buffer tiny so we don't accumulate latency (may be backend-dependent)
#         video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#         # Report actual negotiated values
#         actual_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#         actual_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         actual_fps = video.get(cv2.CAP_PROP_FPS)
#         print(f"[INFO] Capture at {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
#         # --------------------------------------------------

#         output = None
#         if output_stream:
#             width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#             height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             fps = int(video.get(cv2.CAP_PROP_FPS)) or 30
#             codec = cv2.VideoWriter_fourcc(*'XVID')
#             if type(stream) != int:
#                 output_path = os.path.join('..', 'media', 'output', os.path.splitext(os.path.basename(str(stream)))[0] + '.avi')
#             else:
#                 output_path = os.path.join('..', 'media', 'output', f'camera_{stream}.avi')
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)
#             output = cv2.VideoWriter(output_path, codec, fps, (width, height))

#         frame_i = 0  # ---------- ADDED: for throttling prints ----------

#         while video.isOpened():
#             success, frame = video.read()
#             if not success:
#                 break

#             # Run detection (no verbose prints, no built-in annotations)
#             result, overlay = apriltag.detect_tags(
#                 frame,
#                 detector,
#                 camera_params=camera_params,
#                 tag_size=tag_size_m,
#                 vizualization=0,   # keep the overlay clean; we'll draw our own
#                 verbose=0,
#                 annotation=False
#             )

#             # Collect centers and poses for tag 20 and 21, if found
#             centers = {}
#             tvecs = {}
#             for i in range(0, len(result), 4):
#                 det = result[i]        # apriltag.Detection
#                 pose = result[i + 1]   # 3x4 pose matrix (tag->camera)
#                 tid = getattr(det, 'tag_id', None)
#                 if tid in want_ids:
#                     centers[tid] = np.array(det.center, dtype=float)
#                     tvecs[tid] = np.array(pose[:3, 3], dtype=float)

#             # If both present, compute distances and draw
#             if 20 in centers and 21 in centers:
#                 # 2D pixel distance
#                 pix_dist = float(np.linalg.norm(centers[20] - centers[21]))

#                 # 3D distance in meters (using tag poses in camera frame)
#                 if 20 in tvecs and 21 in tvecs:
#                     m_dist = float(np.linalg.norm(tvecs[20] - tvecs[21]))
#                 else:
#                     m_dist = None

#                 # Draw a line between the tag centers
#                 p0 = tuple(np.round(centers[20]).astype(int))
#                 p1 = tuple(np.round(centers[21]).astype(int))
#                 cv2.line(overlay, p0, p1, (0, 255, 255), 2)

#                 # Prepare the text
#                 if m_dist is not None:
#                     text = f"20↔21: {pix_dist:.1f}px | {m_dist:.3f} m"
#                 else:
#                     text = f"20↔21: {pix_dist:.1f}px"

#                 # Put the text near the midpoint of the segment
#                 mid = (int((p0[0] + p1[0]) / 2), int((p0[1] + p1[1]) / 2) - 10)
#                 cv2.putText(overlay, text, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

#                 # ---------- CHANGED: throttle terminal prints ----------
#                 if frame_i % 10 == 0:
#                     print(text)
#                 # ------------------------------------------------------

#             frame_i += 1  # ---------- ADDED ----------

#             # Optionally write/display
#             if output_stream and output is not None:
#                 output.write(overlay)

#             if display_stream:
#                 cv2.imshow(detection_window_name, overlay)
#                 # Press space bar to terminate
#                 if cv2.waitKey(1) & 0xFF == ord(' '):
#                     break

#         if output_stream and output is not None:
#             output.release()
#         video.release()

# ################################################################################

# if __name__ == '__main__':
#     apriltag_video()












#!/usr/bin/env python

from argparse import ArgumentParser
import os
import cv2
import numpy as np
import apriltag
# >>> ADDED: for timestamps & CSV
import csv
import time
from datetime import datetime

################################################################################

def apriltag_video(input_streams=[0],
                   output_stream=False,
                   display_stream=True,
                   detection_window_name='AprilTag'):
    """
    Detect ONLY tags 20 and 21 from video. If both are visible,
    draw a line between their image centers and display:
      - pixel distance between centers
      - 3D distance between tag centers (in meters), using pose estimation
    """

    parser = ArgumentParser(description='Detect AprilTags from video stream.')
    apriltag.add_arguments(parser)
    options = parser.parse_args()

    # Set up a reasonable search path for the apriltag DLL/so/dylib.
    detector = apriltag.Detector(options, searchpath=apriltag._get_dll_path())

    # Hard-coded camera intrinsics and tag size (in meters)
    camera_params = (1408.421651570743, 1405.3445689921414, 1028.1372748266583, 539.4602383823626)
    # fx, fy, cx, cy
    tag_size_m = 0.049  # meters

    want_ids = {20, 21}

    for stream in input_streams:
        video = cv2.VideoCapture(stream)

        # Force 1080p (as before)
        video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # >>> ADDED: helpful smoothness defaults (safe to keep)
        video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        video.set(cv2.CAP_PROP_FPS, 30)
        video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        actual_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = video.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] Capture at {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

        # >>> ADDED: CSV accumulation & timing anchors
        rows = []                 # each row: [timestamp_ms, elapsed_ms, pix_dist, m_dist]
        t0_ns = None              # first-data high-res timestamp (perf counter)
        # CSV filename with start time (wall clock)
        csv_name = f"distance_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        output = None
        if output_stream:
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(video.get(cv2.CAP_PROP_FPS)) or 30
            codec = cv2.VideoWriter_fourcc(*'XVID')
            if type(stream) != int:
                output_path = os.path.join('..', 'media', 'output', os.path.splitext(os.path.basename(str(stream)))[0] + '.avi')
            else:
                output_path = os.path.join('..', 'media', 'output', f'camera_{stream}.avi')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output = cv2.VideoWriter(output_path, codec, fps, (width, height))

        # (keep optional print throttling if you need it)
        frame_i = 0

        while video.isOpened():
            success, frame = video.read()
            if not success:
                break

            # Run detection (no verbose prints, no built-in annotations)
            result, overlay = apriltag.detect_tags(
                frame,
                detector,
                camera_params=camera_params,
                tag_size=tag_size_m,
                vizualization=0,   # keep the overlay clean; we'll draw our own
                verbose=0,
                annotation=False
            )

            # Collect centers and poses for tag 20 and 21, if found
            centers = {}
            tvecs = {}
            for i in range(0, len(result), 4):
                det = result[i]        # apriltag.Detection
                pose = result[i + 1]   # 3x4 pose matrix (tag->camera)
                tid = getattr(det, 'tag_id', None)
                if tid in want_ids:
                    centers[tid] = np.array(det.center, dtype=float)
                    tvecs[tid] = np.array(pose[:3, 3], dtype=float)

            # If both present, compute distances and draw
            if 20 in centers and 21 in centers:
                # 2D pixel distance
                pix_dist = float(np.linalg.norm(centers[20] - centers[21]))

                # 3D distance in meters (using tag poses in camera frame)
                if 20 in tvecs and 21 in tvecs:
                    m_dist = float(np.linalg.norm(tvecs[20] - tvecs[21]))
                else:
                    m_dist = None

                # Draw a line between the tag centers
                p0 = tuple(np.round(centers[20]).astype(int))
                p1 = tuple(np.round(centers[21]).astype(int))
                cv2.line(overlay, p0, p1, (0, 255, 255), 2)

                # Prepare the text
                if m_dist is not None:
                    text = f"20↔21: {pix_dist:.1f}px | {m_dist:.3f} m"
                else:
                    text = f"20↔21: {pix_dist:.1f}px"

                # >>> ADDED: timestamps (real-world & elapsed since first data)
                now = datetime.now()
                ts_ms = f"{now:%Y-%m-%d %H:%M:%S}.{int(now.microsecond/1000):03d}"  # real-world to ms
                now_ns = time.perf_counter_ns()
                if t0_ns is None:
                    t0_ns = now_ns
                elapsed_ms = (now_ns - t0_ns) // 1_000_000  # integer milliseconds

                # Append timestamp to terminal print line
                text_with_time = f"{text} | {ts_ms}"
                # (If you wish to see relative time too, you could add: f" | +{elapsed_ms} ms")

                # Put the text near the midpoint of the segment
                mid = (int((p0[0] + p1[0]) / 2), int((p0[1] + p1[1]) / 2) - 10)
                cv2.putText(overlay, text_with_time, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Terminal print (you can remove the throttle if you want every frame)
                if frame_i % 10 == 0:
                    print(text_with_time)

                # >>> ADDED: store row for CSV (m_dist blank if None)
                rows.append([ts_ms, int(elapsed_ms), f"{pix_dist:.3f}", "" if m_dist is None else f"{m_dist:.6f}"])

            frame_i += 1

            # Optionally write/display
            if output_stream and output is not None:
                output.write(overlay)

            if display_stream:
                cv2.imshow(detection_window_name, overlay)
                # Press space bar to terminate
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    break

        # >>> ADDED: write CSV when stream ends (if we collected any rows)
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
    apriltag_video()
