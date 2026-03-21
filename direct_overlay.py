#!/usr/bin/env python3
"""
Direct overlay generator using fill JSON files with coordinates.
Generates annotated PNG pages with filled values from fill JSON.
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pypdfium2 as pdfium


def render_page(pdf_path: str, page_num: int, scale: float = 2.0) -> tuple[Image.Image, float, float]:
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf.get_page(page_num - 1)
    w = int(page.get_width() * scale)
    h = int(page.get_height() * scale)
    bitmap = page.render(scale=scale)
    img = bitmap.to_pil()
    return img, page.get_width(), page.get_height()


def inches_to_pt(x: float, y: float) -> tuple[float, float]:
    return x * 72, y * 72


def draw_fill_values(img: Image.Image, fill_data: list, page_height_pt: float, scale: float, font):
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    for item in fill_data:
        if item.get("status") != "filled":
            continue
        value = item.get("value")
        if value is None:
            continue
        if isinstance(value, bool):
            if not value:
                continue
            value = "X"
        
        coords = item.get("coords", {})
        x_in, y_in = coords.get("x", 0), coords.get("y", 0)
        x_pt, y_pt = inches_to_pt(x_in, y_in)
        
        x_px = x_pt * scale
        y_px = (page_height_pt - y_pt) * scale
        
        label = item.get("label", "")[:40]
        value_str = str(value)
        
        draw.text((x_px, y_px), f"{label}: {value_str}", fill=(0, 51, 139), font=font)
        print(f"  [{item.get('page')}] {item.get('field_id')} = {value}")


def main():
    import sys
    if len(sys.argv) < 4:
        print("Usage: python direct_overlay.py <pdf_path> <fill_json_path> <output_dir>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    fill_json = sys.argv[2]
    output_dir = Path(sys.argv[3])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(fill_json) as f:
        fill_data = json.load(f)
    
    fill_by_page = {}
    for item in fill_data:
        p = item.get("page", 1)
        fill_by_page.setdefault(p, []).append(item)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 14)
        except:
            font = ImageFont.load_default()
    
    pdf = pdfium.PdfDocument(pdf_path)
    num_pages = len(pdf)
    
    for page_num in range(1, num_pages + 1):
        print(f"\nRendering page {page_num}...")
        img, page_width_pt, page_height_pt = render_page(pdf_path, page_num)
        
        page_fill = fill_by_page.get(page_num, [])
        if page_fill:
            draw_fill_values(img, page_fill, page_height_pt, 2.0, font)
        
        out_path = output_dir / f"f1040_page_{page_num}.png"
        img.save(out_path)
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
