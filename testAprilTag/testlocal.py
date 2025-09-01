import os
import glob
import cv2
from pupil_apriltags import Detector

# === 1) 配置：文件夹与基名（不带扩展名） ===
base_dir = r"D:\Github\AprilTag\testAprilTag"   # 修改为你的目录
base_name = "testTag36_11_00076"                # 不带扩展名

# === 2) 自动寻找实际存在的图像文件 ===
candidates = [
    os.path.join(base_dir, base_name + ext)
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
]
img_path = next((p for p in candidates if os.path.exists(p)), None)

if img_path is None:
    # 兜底：也可以在目录里自动找第一个 svg->png/jpg 的文件
    all_imgs = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        all_imgs.extend(glob.glob(os.path.join(base_dir, ext)))
    raise FileNotFoundError(
        "找不到目标图像。\n"
        f"已尝试：\n  " + "\n  ".join(candidates) + "\n"
        "该目录下找到的图片有：\n  " + "\n  ".join(all_imgs or ["<空>"])
    )

print(f"[INFO] 使用图像: {img_path}")

# === 3) 读取为灰度图（确保得到 uint8 的 2D 数组） ===
gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if gray is None:
    raise RuntimeError(
        f"OpenCV 无法读取图像：{img_path}\n"
        "可能原因：路径包含转义符、没有读取权限、文件损坏、或为 SVG 未栅格化。"
    )

# === 4) 构造检测器（家族名要与生成的标签一致，常用 tag36h11） ===
detector = Detector(
    families="tag36h11",  # 如果你的标签是 36h11
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=True,
)

# === 5) 检测 ===
detections = detector.detect(gray)
print(f"[INFO] Found {len(detections)} tags")

# === 6) 可视化 ===
vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
for d in detections:
    c = d.corners.astype(int)
    cv2.polylines(vis, [c], True, (0, 255, 0), 2)
    cx, cy = map(int, d.center)
    cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
    cv2.putText(vis, f"id:{d.tag_id}", (c[0][0], c[0][1]-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

cv2.imshow("detections", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
