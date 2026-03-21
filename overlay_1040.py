#!/usr/bin/env python3
"""
Generate overlays for 1040 with only inferred cell values printed.
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pypdfium2 as pdfium

DATA_DIR = Path("/home/sweeden/projects/docling_data/tax_packet")
OUT_DIR = DATA_DIR / "annotated"
OUT_DIR.mkdir(exist_ok=True)

def render_page(pdf_path, page_num, scale=2.0):
    pdf = pdfium.PdfDocument(str(pdf_path))
    page = pdf[page_num]
    bitmap = page.render(scale=scale)
    return bitmap.to_pil()

def create_overlay(fields, page_num, w, h, scale=2.0):
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", int(10*scale))
    except:
        font = ImageFont.load_default()
    for f in fields:
        if f.get("page") != page_num:
            continue
        x = int(f["coords"]["x"] * 72 * scale)
        y = int(f["coords"]["y"] * 72 * scale)
        draw.rectangle([x, y, x+100, y+20], outline='red', width=1)
        draw.text((x+2, y+2), f.get("field_id", "")[:15], fill='red', font=font)
        if f.get("value"):
            draw.text((x+105, y+2), f"= {f['value']}", fill='blue', font=font)
    return overlay

def main():
    fill_file = DATA_DIR / "f1040_fill.json"
    if not fill_file.exists():
        print("Fill file not found")
        return
    
    with open(fill_file) as f:
        instructions = json.load(f)
    
    pdf = DATA_DIR / "f1040.pdf"
    if not pdf.exists():
        print("PDF not found")
        return
    
    num_pages = len(pdfium.PdfDocument(str(pdf)))
    
    for page_num in range(1, num_pages + 1):
        page_img = render_page(pdf, page_num - 1)
        overlay = create_overlay(instructions, page_num, page_img.width, page_img.height)
        
        if overlay.size != page_img.size:
            overlay = overlay.resize(page_img.size, Image.LANCZOS)
        
        merged = Image.alpha_composite(page_img.convert('RGBA'), overlay).convert('RGB')
        out_path = OUT_DIR / f"f1040_page{page_num}.png"
        merged.save(out_path)
        
        for f in instructions:
            if f.get("page") == page_num and f.get("status") == "filled" and f.get("value"):
                print(f"[{page_num}] {f['field_id']} = {f['value']}")

if __name__ == "__main__":
    main()
