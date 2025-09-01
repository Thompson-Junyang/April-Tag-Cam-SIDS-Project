import os
import glob
import cairosvg

# 单文件示例 —— 注意用 r'' 或者把 \ 改成 /
svg_path = r"D:\Github\AprilTag\testAprilTag\tag36_11_00076.svg"
out_path = r"D:\Github\AprilTag\testAprilTag\testTag36_11_00076.png"
cairosvg.svg2png(url=svg_path, write_to=out_path, dpi=300)

print("Single file converted:", out_path)

# 批量：把整个文件夹下的 .svg 转为 .png
in_dir  = r"D:\Github\AprilTag\testAprilTag"
out_dir = r"D:\Github\AprilTag\testAprilTag\png_out"
os.makedirs(out_dir, exist_ok=True)

for svg in glob.glob(os.path.join(in_dir, "*.svg")):
    base = os.path.splitext(os.path.basename(svg))[0]
    png  = os.path.join(out_dir, base + ".png")
    cairosvg.svg2png(url=svg, write_to=png, dpi=300)
    print("Converted:", png)

print("Done.")
