#!/usr/bin/env python3
"""
IRS Form 1040 analysis and fill pipeline.
Minimal print - only prints inferred cell values.
"""

import json
import sys
from pathlib import Path

DATA_DIR = Path("/home/sweeden/projects/docling_data")
FORM_DIR = DATA_DIR / "tax_packet"
OUT_DIR = FORM_DIR
OUT_DIR.mkdir(exist_ok=True)

PDF_1040 = FORM_DIR / "f1040.pdf"
SCHEMA_FILE = OUT_DIR / "f1040_schema.json"
DATA_FILE = OUT_DIR / "f1040_data.json"
FILL_FILE = OUT_DIR / "f1040_fill.json"

def get_path(obj, dotpath):
    if not dotpath:
        return None
    parts = []
    for seg in dotpath.replace("][", ".").replace("[", ".").replace("]", "").split("."):
        parts.append(int(seg) if seg.lstrip("-").isdigit() else seg)
    cur = obj
    for p in parts:
        if cur is None:
            return None
        cur = cur[p] if isinstance(p, int) and isinstance(cur, list) else cur.get(p) if isinstance(cur, dict) else None
    return cur

def resolve_value(field_def, data):
    dp = field_def.get("data_path")
    fv = field_def.get("fill_value")
    resolved = get_path(data, dp) if dp else None
    if resolved is None:
        resolved = fv
    if resolved is None or resolved == "":
        return ("", "skip") if field_def.get("field_type") != "checkbox" else (False, "skip")
    return (resolved, "filled")

def build_fill(schema, data):
    instructions = []
    for page_key, page_data in schema.get("pages", {}).items():
        if not isinstance(page_data, dict):
            continue
        page_num = page_data.get("pdf_page", 1)
        for fd in page_data.get("fields", []):
            val, status = resolve_value(fd, data)
            instructions.append({
                "page": page_num,
                "field_id": fd.get("field_id", ""),
                "label": fd.get("label", ""),
                "value": val,
                "status": status,
                "coords": fd.get("coords", {}),
            })
    return instructions

def main():
    if not SCHEMA_FILE.exists():
        print(f"Schema not found: {SCHEMA_FILE}")
        sys.exit(1)
    if not DATA_FILE.exists():
        print(f"Data not found: {DATA_FILE}")
        sys.exit(1)

    with open(SCHEMA_FILE) as f:
        schema = json.load(f)
    with open(DATA_FILE) as f:
        data = json.load(f)

    instructions = build_fill(schema, data)
    
    with open(FILL_FILE, "w") as f:
        json.dump(instructions, f, indent=2)

    filled = sum(1 for i in instructions if i["status"] == "filled")
    
    for item in instructions:
        if item["status"] == "filled":
            print(f"[{item['page']}] {item['field_id']} = {item['value']}")

if __name__ == "__main__":
    main()
