"""
AprilTag multi-tag live pose & pairwise distance (Windows, undistorted pipeline)

Usage:
  1) pip install opencv-python pupil-apriltags numpy
  2) Set TAG_SIZE_M (meters) & TAG_FAMILY as needed
  3) python apriltag_live_dist_undistorted.py
Keys:
  - 's': print pairwise distances in console
  - 'e': export NxN CSV distance matrix (ids as headers)
  - 'q': quit
"""

import cv2
import numpy as np
from itertools import combinations
import csv
import time
import os

# ========= 1) Camera intrinsics (from user, non-fisheye) =========
FX = 1433.0824878559827
FY = 1435.1336226361313
CX = 943.793605483019
CY = 546.6106650505985
CAM_MTX = np.array([[FX, 0.0, CX],
                    [0.0, FY, CY],
                    [0.0, 0.0, 1.0]], dtype=np.float64)

DIST_COEFFS = np.array([[-0.7726239078491605,
                          1.1706904215044034,
                          0.000712858225255037,
                          0.001023401492418749,
                          -1.2195922716077368]], dtype=np.float64)

# ========= 2) AprilTag parameters =========
TAG_SIZE_M = 0.035            # <-- set your physical tag edge length in meters
TAG_FAMILY = "tag36h11"       # e.g., 'tag36h11', 'tag25h9', etc.

# ========= 3) Viz parameters =========
AXIS_LEN = TAG_SIZE_M * 0.5   # meters, axis length drawn from tag center
FONT = cv2.FONT_HERSHEY_SIMPLEX


def ensure_detector():
    """Create a pupil_apriltags Detector."""
    try:
        from pupil_apriltags import Detector
    except Exception as e:
        raise RuntimeError(
            "pupil-apriltags not installed.\n"
            "Install with: pip install pupil-apriltags\n"
            f"Detail: {e}"
        )
    return Detector(
        families=TAG_FAMILY,
        nthreads=1,
        quad_decimate=1.0,     # speed/accuracy tradeoff: increase to 2~3 for speed
        quad_sigma=0.0,
        refine_edges=True,
        decode_sharpening=0.25,
        debug=False
    )


def build_undistort_maps(K, D, size_wh):
    """
    Precompute undistortion maps and new intrinsics for rectified (undistorted) image.

    Returns:
        K_new: 3x3 new camera matrix after undistortion
        map1, map2: remap maps for cv2.remap
    """
    K_new, _ = cv2.getOptimalNewCameraMatrix(K, D, size_wh, alpha=0)  # crop to valid region
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K_new, size_wh, cv2.CV_16SC2)
    return K_new, map1, map2


def project_points_no_dist(obj_pts, rvec, tvec, K):
    """Project 3D points using zero distortion (consistent with undistorted image)."""
    dist0 = np.zeros((1, 5), dtype=np.float64)
    img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist0)
    return img_pts


def draw_axes(img, rvec, tvec, K, length=0.05, thickness=2):
    """Draw right-handed axes (x=red,y=green,z=blue) centered at tag origin."""
    axis_3d = np.float32([[0, 0, 0],
                          [length, 0, 0],
                          [0, length, 0],
                          [0, 0, length]])
    pts = project_points_no_dist(axis_3d, rvec, tvec, K)
    p0, px, py, pz = [tuple(np.int32(p.ravel())) for p in pts]
    cv2.line(img, p0, px, (0, 0, 255), thickness)   # X - red
    cv2.line(img, p0, py, (0, 255, 0), thickness)   # Y - green
    cv2.line(img, p0, pz, (255, 0, 0), thickness)   # Z - blue


def export_distance_csv(ids, tvecs, out_dir=".", prefix="apriltag_dist"):
    """
    Save an NxN symmetric distance matrix (meters) with tag ids as headers.
    Only ids present in `ids` will be used; tvecs[id] are 3D positions in camera frame.
    """
    if not ids:
        print("[WARN] No tags to export.")
        return None
    # build matrix
    N = len(ids)
    M = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i == j:
                M[i, j] = 0.0
            else:
                M[i, j] = float(np.linalg.norm(tvecs[ids[i]] - tvecs[ids[j]]))

    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{prefix}_{stamp}.csv")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id\\id"] + ids)
        for i in range(N):
            row = [ids[i]] + [f"{M[i, j]:.6f}" for j in range(N)]
            writer.writerow(row)

    print(f"[INFO] Saved distance matrix to: {path}")
    return path


def main():
    # Open Windows camera index 0
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)  # fallback
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera 0.")

    # Read one frame to get size
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Failed to grab initial frame from camera.")
    h, w = frame0.shape[:2]

    # Precompute undistortion maps and new intrinsics
    K_new, map1, map2 = build_undistort_maps(CAM_MTX, DIST_COEFFS, (w, h))

    # Create detector
    detector = ensure_detector()
    print("[INFO] Using pupil_apriltags on undistorted frames.")
    print("[INFO] Press 's' to print distances, 'e' to export CSV, 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to read frame.")
            break

        # Undistort the frame
        undist = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

        # Detect on undistorted image; use K_new (fx,fy,cx,cy). Distortion is implicitly 0.
        detections = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=[K_new[0, 0], K_new[1, 1], K_new[0, 2], K_new[1, 2]],
            tag_size=TAG_SIZE_M
        )

        # Collect poses (in camera frame)
        tvecs = {}         # id -> (3,)
        rvecs = {}         # id -> (3,)
        centers_2d = {}    # id -> (u, v) on undistorted image

        for det in detections:
            tag_id = int(det.tag_id)
            R = det.pose_R               # 3x3 rotation (tag -> camera)
            t = det.pose_t.reshape(3)    # translation in meters (tag origin in camera frame)
            rvec, _ = cv2.Rodrigues(R)

            # draw quad
            corners = det.corners.astype(int)  # 4x2 in undistorted image
            for i in range(4):
                cv2.line(undist, tuple(corners[i]),
                         tuple(corners[(i + 1) % 4]),
                         (0, 255, 255), 2)

            cX, cY = map(int, det.center)
            cv2.circle(undist, (cX, cY), 4, (0, 0, 255), -1)

            tvecs[tag_id] = t
            rvecs[tag_id] = rvec
            centers_2d[tag_id] = (cX, cY)

            # annotate id & z
            z_m = float(t[2])
            cv2.putText(undist, f"id:{tag_id} z:{z_m:.3f}m",
                        (cX + 6, cY - 6), FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(undist, f"id:{tag_id} z:{z_m:.3f}m",
                        (cX + 6, cY - 6), FONT, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

            # draw axes centered at tag origin (consistent with undistorted model)
            draw_axes(undist, rvec, t, K_new, length=AXIS_LEN, thickness=2)

        # Pairwise distances in camera frame
        ids = sorted(tvecs.keys())
        distances = []
        for i, j in combinations(ids, 2):
            d = float(np.linalg.norm(tvecs[i] - tvecs[j]))
            distances.append((i, j, d))
            # optional: draw connecting line with distance label
            if i in centers_2d and j in centers_2d:
                p1, p2 = centers_2d[i], centers_2d[j]
                cv2.line(undist, p1, p2, (200, 200, 200), 1)
                mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                txt = f"{d:.3f}m"
                cv2.putText(undist, txt, mid, FONT, 0.5, (50, 50, 50), 2, cv2.LINE_AA)
                cv2.putText(undist, txt, (mid[0] + 1, mid[1] + 1),
                            FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # HUD
        cv2.putText(undist, f"Tags: {len(ids)}",
                    (20, 35), FONT, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(undist, f"Tags: {len(ids)}",
                    (20, 35), FONT, 1.0, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("AprilTag Live - Distances (undistorted)", undist)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            if distances:
                print("\n=== Pairwise distances (meters) ===")
                for (i, j, d) in distances:
                    print(f"tag {i} <-> tag {j}: {d:.4f} m")
            else:
                print("\n[INFO] No pairwise distances available (0/1 tag).")
        elif key == ord('e'):
            out = export_distance_csv(ids, tvecs, out_dir=".", prefix="apriltag_dist")
            if out:
                cv2.displayStatusBar("AprilTag Live - Distances (undistorted)",
                                     f"Saved: {out}", 2000)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
