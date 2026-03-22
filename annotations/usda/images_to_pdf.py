#!/usr/bin/env python3
"""
Combine a list of images into a single PDF with each image centered and scaled
to fit the entire page (letter size). Uses higher internal resolution for
crisp text when viewing or printing.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import List, Tuple

from PIL import Image

# US Letter in points (1/72 inch) - PDF page size
PAGE_WIDTH_PT = 612
PAGE_HEIGHT_PT = 792

# Render at 2x resolution (144 DPI equivalent) for crisper text; use 3 for even sharper
RESOLUTION_SCALE = 2


def scale_to_fit(img_w: int, img_h: int, page_w: int, page_h: int) -> Tuple[float, int, int]:
    """Return (scale, scaled_w, scaled_h) so image fits within page."""
    scale = min(page_w / img_w, page_h / img_h)
    return scale, int(round(img_w * scale)), int(round(img_h * scale))


def build_high_res_page(img: Image.Image, scale: int = RESOLUTION_SCALE) -> Image.Image:
    """Build a high-res image: letter-size content at scale*72 DPI, centered on white."""
    page_w = PAGE_WIDTH_PT * scale
    page_h = PAGE_HEIGHT_PT * scale
    img_w, img_h = img.size
    if img.mode != "RGB":
        img = img.convert("RGB")
    _, scaled_w, scaled_h = scale_to_fit(img_w, img_h, page_w, page_h)
    resized = img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
    page = Image.new("RGB", (int(page_w), int(page_h)), (255, 255, 255))
    x = (int(page_w) - scaled_w) // 2
    y = (int(page_h) - scaled_h) // 2
    page.paste(resized, (x, y))
    return page


def main() -> None:
    annotations_dir = Path(__file__).resolve().parent
    annotated_dir = annotations_dir / "annotated"

    # Order: page_1 through page_14, with page_4 as the base image (not annotated)
    image_paths = [
        annotated_dir / "page_1_multi_annotated.png",
        annotated_dir / "page_2_multi_annotated.png",
        annotated_dir / "page_3_multi_annotated.png",
        annotations_dir / "page_4.png",  # base image per user request
        annotated_dir / "page_5_multi_annotated.png",
        annotated_dir / "page_6_multi_annotated.png",
        annotated_dir / "page_7_multi_annotated.png",
        annotated_dir / "page_8_multi_annotated.png",
        annotated_dir / "page_9_multi_annotated.png",
        annotated_dir / "page_10_multi_annotated.png",
        annotated_dir / "page_11_multi_annotated.png",
        annotated_dir / "page_12_multi_annotated.png",
        annotated_dir / "page_13_multi_annotated.png",
        annotated_dir / "page_14_multi_annotated.png",
    ]

    for path in image_paths:
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

    out_path = annotations_dir / "FSA2001_annotated_pages.pdf"

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas
    except ImportError:
        # Fallback: Pillow-only at higher resolution (page size may not be letter in viewer)
        pages: List[Image.Image] = []
        for path in image_paths:
            img = Image.open(path)
            page_img = build_high_res_page(img)
            pages.append(page_img)
        pages[0].save(
            str(out_path),
            "PDF",
            save_all=True,
            append_images=pages[1:],
        )
        print(
            f"Saved: {out_path} (Pillow fallback, {RESOLUTION_SCALE}x resolution). "
            "Install reportlab for letter-size pages: pip install reportlab"
        )
        return

    # ReportLab path: letter-size page with high-res embedded image → crisp at any zoom/print
    c = canvas.Canvas(str(out_path), pagesize=letter)
    for path in image_paths:
        img = Image.open(path)
        page_img = build_high_res_page(img)
        buf = io.BytesIO()
        page_img.save(buf, format="PNG")
        buf.seek(0)
        c.drawImage(
            ImageReader(buf),
            0,
            0,
            width=PAGE_WIDTH_PT,
            height=PAGE_HEIGHT_PT,
            preserveAspectRatio=True,
            anchor="c",
        )
        c.showPage()
    c.save()
    print(f"Saved: {out_path} (resolution scale={RESOLUTION_SCALE}x, ~{72 * RESOLUTION_SCALE} DPI)")


if __name__ == "__main__":
    main()
