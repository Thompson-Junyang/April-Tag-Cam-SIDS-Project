"""
Generate an A4 sheet of AprilTag tag36h11 with each tag sized to 3 cm.

Requirements:
  pip install pillow requests reportlab

Notes:
- Uses official pre-rendered tag36h11 PNGs from AprilRobotics/apriltag-imgs.
- Saves both a high-DPI PNG and a vector-ish PDF (raster placed) for printing.

"""

import io
import math
import os
from typing import List
import requests
from PIL import Image, ImageDraw, ImageFont

# ================== User knobs ==================
TAG_IDS: List[int] = list(range(0, 24))  # ← 你要打印的 tag ID（0–586）多了会自动分页
TAG_SIZE_MM = 20.0                       # 每个标签黑边到黑边的目标尺寸（毫米）= 3 cm
MARGIN_MM = 10.0                         # 版心外边距（毫米）
GAP_MM = 6.0                             # 标签之间最小间距（毫米）
ANNOTATE_ID = True                       # 标签下方是否打印“ID=xxx”
FONT_SIZE_PT = 9                         # 注记字号（点）

# 输出文件名（会自动加页码）
OUT_BASENAME = f"apriltag36h11_A4_{TAG_SIZE_MM}mm"


# 纸张与分辨率
A4_W_MM, A4_H_MM = 210.0, 297.0
DPI = 300  # 推荐 300dpi 打印

# 官方图集 raw URL 模板（ID 用 5 位零填充）
RAW_URL_TMPL = "https://raw.githubusercontent.com/AprilRobotics/apriltag-imgs/master/tag36h11/tag36_11_{id:05d}.png"

# =================================================

def mm_to_px(mm: float, dpi: int = DPI) -> int:
    return int(round(mm * dpi / 25.4))

def fetch_tag_png(tag_id: int) -> Image.Image:
    url = RAW_URL_TMPL.format(id=tag_id)
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("L")  # 官方是黑白小图
    # 官方图片是极小像素（每码元1px），需放大到我们需要的物理尺寸
    return img

def ensure_font(size_pt: int):
    # 尽量找一个系统字体；若失败，用 PIL 默认位图字体
    try:
        # 常见可用字体名（Windows/Ubuntu/macOS 里通常有）
        for name in ["Arial.ttf", "DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"]:
            if os.path.exists(name):
                return ImageFont.truetype(name, size=size_pt)
        # 尝试 Mac/Linux 常见路径
        for path in [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]:
            if os.path.exists(path):
                return ImageFont.truetype(path, size=size_pt)
    except Exception:
        pass
    return ImageFont.load_default()

def layout_grid_per_page(tag_px: int, margin_px: int, gap_px: int, page_w_px: int, page_h_px: int, annotate_h_px: int):
    """计算一页能排多少列/行，以及每页的起始坐标网格。"""
    # 每个单元高度需要考虑标签图像高度 + 注记高度（若开启）
    cell_w = tag_px
    cell_h = tag_px + (annotate_h_px if ANNOTATE_ID else 0)

    usable_w = page_w_px - 2 * margin_px
    usable_h = page_h_px - 2 * margin_px

    # 至少放 1 列/行
    cols = max(1, (usable_w + gap_px) // (cell_w + gap_px))
    rows = max(1, (usable_h + gap_px) // (cell_h + gap_px))

    # 实际占用尺寸
    total_w = cols * cell_w + (cols - 1) * gap_px
    total_h = rows * cell_h + (rows - 1) * gap_px

    # 左上角起点（居中）
    start_x = margin_px + (usable_w - total_w) // 2
    start_y = margin_px + (usable_h - total_h) // 2

    # 生成网格左上角坐标
    slots = []
    y = start_y
    for _ in range(rows):
        x = start_x
        for _ in range(cols):
            slots.append((x, y))
            x += cell_w + gap_px
        y += cell_h + gap_px

    return cols, rows, slots

def render_pages(tag_ids: List[int]):
    # 计算像素尺寸
    page_w_px = mm_to_px(A4_W_MM)
    page_h_px = mm_to_px(A4_H_MM)
    margin_px = mm_to_px(MARGIN_MM)
    gap_px = mm_to_px(GAP_MM)
    tag_px = mm_to_px(TAG_SIZE_MM)

    # 字体和注记高度估计
    font = ensure_font(FONT_SIZE_PT)
    annotate_h_px = int(round(FONT_SIZE_PT * DPI / 72.0 * 1.6)) if ANNOTATE_ID else 0

    # 网格布局
    cols, rows, slots = layout_grid_per_page(tag_px, margin_px, gap_px, page_w_px, page_h_px, annotate_h_px)
    per_page = cols * rows
    if per_page <= 0:
        raise RuntimeError("参数导致一页容纳数量为 0，请增大页边距/减小标签/间距。")

    pages = []
    # 分页
    for p in range(0, len(tag_ids), per_page):
        this_ids = tag_ids[p:p+per_page]
        canvas = Image.new("L", (page_w_px, page_h_px), 255)  # 白底
        draw = ImageDraw.Draw(canvas)

        for (x, y), tid in zip(slots, this_ids):
            # 下载 + 尺寸缩放（最近邻放大，保证边界清晰）
            base = fetch_tag_png(tid)
            tag = base.resize((tag_px, tag_px), resample=Image.NEAREST)

            canvas.paste(tag, (x, y))

            if ANNOTATE_ID:
                text = f"ID={tid}"
                # tw, th = draw.textsize(text, font=font)
                try:
                    # Pillow ≥ 10.0 推荐用 textbbox
                    bbox = draw.textbbox((0, 0), text, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except AttributeError:
                    # 兼容旧 Pillow
                    tw, th = draw.textsize(text, font=font)

                tx = x + (tag_px - tw) // 2
                ty = y + tag_px + max(0, (annotate_h_px - th)//2)
                draw.text((tx, ty), text, font=font, fill=0)

        pages.append(canvas)

    return pages

def save_png_and_pdf(pages: List[Image.Image], basename: str):
    # PNG
    for i, im in enumerate(pages, start=1):
        fn = f"{basename}_p{i}.png"
        im.save(fn, dpi=(DPI, DPI))
        print("Saved:", fn)

    # PDF（多页）
    pdf_fn = f"{basename}.pdf"
    rgb_pages = [p.convert("RGB") for p in pages]
    if len(rgb_pages) == 1:
        rgb_pages[0].save(pdf_fn, save_all=False)
    else:
        rgb_pages[0].save(pdf_fn, save_all=True, append_images=rgb_pages[1:])
    print("Saved:", pdf_fn)

if __name__ == "__main__":
    pages = render_pages(TAG_IDS)
    save_png_and_pdf(pages, OUT_BASENAME)
    print("Done. Print at 100% scale. Each tag edge length =", TAG_SIZE_MM, "mm")
