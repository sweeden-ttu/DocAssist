#!/usr/bin/env python3
"""
Generate overlays for tax forms - only prints inferred values.
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pypdfium2 as pdfium

DATA_DIR = Path("/home/sweeden/projects/docling_data/tax_packet")
OUT_DIR = DATA_DIR / "annotated"
OUT_DIR.mkdir(exist_ok=True)

def get_path(obj, dotpath):
    if not dotpath: return None
    parts = []
    for seg in dotpath.replace("][", ".").replace("[", ".").replace("]", "").split("."):
        parts.append(int(seg) if seg.lstrip("-").isdigit() else seg)
    cur = obj
    for p in parts:
        if cur is None: return None
        cur = cur[p] if isinstance(p, int) and isinstance(cur, list) else cur.get(p) if isinstance(cur, dict) else None
    return cur

def resolve_value(fd, data):
    dp = fd.get("data_path")
    fv = fd.get("fill_value")
    resolved = get_path(data, dp) if dp else None
    if resolved is None: resolved = fv
    if resolved is None or resolved == "": return ("", "skip") if fd.get("field_type") != "checkbox" else (False, "skip")
    return (resolved, "filled")

def build_fill(schema, data):
    instructions = []
    for pk, pd in schema.get("pages", {}).items():
        if not isinstance(pd, dict): continue
        pn = pd.get("pdf_page", 1)
        for fd in pd.get("fields", []):
            val, status = resolve_value(fd, data)
            instructions.append({"page": pn, "field_id": fd.get("field_id", ""), "value": val, "status": status, "coords": fd.get("coords", {})})
    return instructions

def render_page(pdf_path, page_num, scale=2.0):
    pdf = pdfium.PdfDocument(str(pdf_path))
    page = pdf[page_num]
    return page.render(scale=scale).to_pil()

def create_overlay(fields, page_num, w, h, scale=2.0):
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", int(10*scale))
    except: font = ImageFont.load_default()
    for f in fields:
        if f.get("page") != page_num: continue
        x, y = int(f["coords"]["x"] * 72 * scale), int(f["coords"]["y"] * 72 * scale)
        draw.rectangle([x, y, x+120, y+18], outline='red', width=1)
        draw.text((x+2, y+2), f.get("field_id", "")[:18], fill='red', font=font)
        if f.get("value"): draw.text((x+125, y+2), f"= {f['value']}", fill='blue', font=font)
    return overlay

def process_form(name, pdf_file, schema_file, data_file):
    schema_path = DATA_DIR / schema_file
    data_path = DATA_DIR / data_file
    if not schema_path.exists() or not data_path.exists():
        print(f"Missing files for {name}")
        return
    
    with open(schema_path) as f: schema = json.load(f)
    with open(data_path) as f: data = json.load(f)
    
    instructions = build_fill(schema, data)
    pdf = DATA_DIR / pdf_file
    if not pdf.exists():
        print(f"PDF not found: {pdf}")
        return
    
    num_pages = len(pdfium.PdfDocument(str(pdf)))
    for page_num in range(1, num_pages + 1):
        page_img = render_page(pdf, page_num - 1)
        overlay = create_overlay(instructions, page_num, page_img.width, page_img.height)
        if overlay.size != page_img.size: overlay = overlay.resize(page_img.size, Image.LANCZOS)
        merged = Image.alpha_composite(page_img.convert('RGBA'), overlay).convert('RGB')
        merged.save(OUT_DIR / f"{name}_page{page_num}.png")
        for f in instructions:
            if f.get("page") == page_num and f.get("status") == "filled" and f.get("value"):
                print(f"[{page_num}] {f['field_id']} = {f['value']}")

if __name__ == "__main__":
    process_form("f1040", "f1040.pdf", "f1040_schema.json", "f1040_data.json")
    print()
    process_form("f1065", "f1065.pdf", "f1065_schema.json", "f1065_data.json")
