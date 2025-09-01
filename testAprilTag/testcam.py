# # 仅检测（无位姿）
# python apriltag_cam.py --device 0 --families tag36h11 --width 1280 --height 720

# # 启用位姿（假定标签边长 4cm，粗估内参）
# python apriltag_cam.py --estimate-pose --tag-size 0.04 --device 0 --width 1280 --height 720

# # 如果你有已知内参（示例）
# python apriltag_cam.py --estimate-pose --tag-size 0.04 \
#   --fx 920.0 --fy 920.0 --cx 640 --cy 360 \
#   --device 0 --width 1280 --height 720




import argparse
import time
import cv2
import numpy as np
from pupil_apriltags import Detector

def parse_args():
    p = argparse.ArgumentParser(description="USB Camera AprilTag Detector (Windows)")
    p.add_argument("--device", type=int, default=0, help="camera index (0,1,2,...)")
    p.add_argument("--width", type=int, default=1280, help="capture width")
    p.add_argument("--height", type=int, default=720, help="capture height")
    p.add_argument("--families", type=str, default="tag36h11",
                   help="tag families, e.g. tag36h11,tag25h9")
    p.add_argument("--decimate", type=float, default=1.0, help="quad_decimate (speed/accuracy)")
    p.add_argument("--threads", type=int, default=1, help="nthreads for detector")
    p.add_argument("--sigma", type=float, default=0.0, help="quad_sigma (blur for detection)")
    p.add_argument("--refine", action="store_true", help="refine edges")
    p.add_argument("--estimate-pose", action="store_true", help="estimate tag pose")
    p.add_argument("--tag-size", type=float, default=0.04,
                   help="tag size in meters (outer black square side length)")
    p.add_argument("--fx", type=float, default=None, help="focal length x (pixels)")
    p.add_argument("--fy", type=float, default=None, help="focal length y (pixels)")
    p.add_argument("--cx", type=float, default=None, help="principal point x (pixels)")
    p.add_argument("--cy", type=float, default=None, help="principal point y (pixels)")
    return p.parse_args()

def draw_axes(img, R, t, K, length=0.03, thickness=2):
    """Draw 3D axes on the image using solvePnP-style pose (R: 3x3, t: 3x1, meters)."""
    # 3D axes points (X red, Y green, Z blue)
    axes_3d = np.float32([[0,0,0],
                          [length,0,0],
                          [0,length,0],
                          [0,0,length]]).reshape(-1,3)
    # Project
    Rt = np.hstack([R, t.reshape(3,1)])
    P = K @ Rt
    pts2d = []
    for X in axes_3d:
        Xh = np.hstack([X, 1.0])
        xh = P @ Xh
        pts2d.append((xh[0]/xh[2], xh[1]/xh[2]))
    o, x, y, z = [tuple(map(int, p)) for p in pts2d]
    cv2.line(img, o, x, (0,0,255), thickness)   # X - red
    cv2.line(img, o, y, (0,255,0), thickness)   # Y - green
    cv2.line(img, o, z, (255,0,0), thickness)   # Z - blue

def main():
    args = parse_args()

    # --- Open camera with DirectShow backend (Windows) ---
    cap = cv2.VideoCapture(args.device, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    # 可选：减少自动曝光/自动白平衡波动（不同相机支持不同）
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 0.25/0.75/1 各品牌不一致
    # cap.set(cv2.CAP_PROP_EXPOSURE, -6)     # 因机型而异

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.device}. "
                           "检查设备索引或摄像头权限。")

    detector = Detector(
        families=args.families,
        nthreads=args.threads,
        quad_decimate=args.decimate,
        quad_sigma=args.sigma,
        refine_edges=args.refine
    )

    # camera params
    K = None
    do_pose = args.estimate_pose
    if do_pose:
        # 如果未提供 fx,fy,cx,cy，使用粗估（基于分辨率）
        fx = args.fx if args.fx is not None else 1000.0
        fy = args.fy if args.fy is not None else 1000.0
        cx = args.cx if args.cx is not None else args.width / 2.0
        cy = args.cy if args.cy is not None else args.height / 2.0
        cam_params = [fx, fy, cx, cy]
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float32)
    else:
        cam_params = None

    prev = time.time()
    fps = 0.0

    print("[INFO] Press 'q' to quit, 's' to save a frame.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to grab frame")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        t0 = time.time()
        if do_pose and cam_params is not None:
            dets = detector.detect(
                gray, estimate_tag_pose=True,
                camera_params=cam_params,
                tag_size=args.tag_size
            )
        else:
            dets = detector.detect(gray)
        t1 = time.time()

        vis = frame.copy()
        for d in dets:
            # corners, center
            c = d.corners.astype(int)
            cv2.polylines(vis, [c], True, (0, 255, 0), 2)
            cx_i, cy_i = map(int, d.center)
            cv2.circle(vis, (cx_i, cy_i), 3, (0, 0, 255), -1)
            cv2.putText(vis, f"id:{d.tag_id}", (c[0][0], c[0][1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # pose (if asked)
            if do_pose and hasattr(d, "pose_R") and hasattr(d, "pose_t") and K is not None:
                R = d.pose_R  # 3x3
                t = d.pose_t  # 3x1
                draw_axes(vis, R, t, K, length=args.tag_size*0.75, thickness=2)
                # 显示距离（Z 轴朝外，单位米）
                dist = float(np.linalg.norm(t))
                cv2.putText(vis, f"{dist:.3f} m", (cx_i+8, cy_i-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # FPS
        now = time.time()
        dt = now - prev
        prev = now
        fps = 0.9*fps + 0.1*(1.0/max(dt, 1e-6))
        cv2.putText(vis, f"{fps:.1f} FPS  ({(t1-t0)*1000:.1f} ms/detect)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 50), 2)

        cv2.imshow("AprilTag", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            out = f"frame_{int(time.time())}.png"
            cv2.imwrite(out, vis)
            print("[INFO] saved", out)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
