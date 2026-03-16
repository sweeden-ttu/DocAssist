#!/usr/bin/env python3
"""
fsa2001_fill_from_schema.py
────────────────────────────────────────────────────────────────────────
Reads:
  - fsa2001_field_mapping_schema.json   (field ↔ data-path mapping)
  - fsa2001_weeden_data_1.json          (applicant data)

Produces:
  - fsa2001_fill_instructions.json      (flat list of fill ops keyed by
                                         field_id, ready for a PDF stamper
                                         or the existing overlay_forms.py)

KDDRFI fields are flagged but not skipped — they appear in the output
with fill_value=None and status="PENDING" so the PDF stamper can warn.

Usage:
    python fsa2001_fill_from_schema.py \
        --schema fsa2001_field_mapping_schema.json \
        --data   fsa2001_weeden_data_1.json \
        --out    fsa2001_fill_instructions.json
"""

import argparse
import json
import sys
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def get_path(obj, dotpath: str):
    """Resolve dot-notation path like 'crop_plan[2].est_acres' into a value."""
    if not dotpath:
        return None
    parts = []
    for segment in dotpath.replace("][", ".").replace("[", ".").replace("]", "").split("."):
        if segment.lstrip("-").isdigit():
            parts.append(int(segment))
        else:
            parts.append(segment)
    cur = obj
    for p in parts:
        if cur is None:
            return None
        if isinstance(p, int):
            if isinstance(cur, list) and 0 <= p < len(cur):
                cur = cur[p]
            else:
                return None
        else:
            cur = cur.get(p) if isinstance(cur, dict) else None
    return cur


def resolve_value(field_def: dict, applicant: dict):
    """
    Determine what value to write into a field.
    Priority: data_path → fill_value → None
    Returns (value, status) where status is:
        "filled"  – ready to write
        "pending" – KDDRFI / needs collection
        "skip"    – leave blank (empty string, or false for checkbox)
    """
    data_path = field_def.get("data_path")
    fill_value = field_def.get("fill_value")

    resolved = None
    if data_path:
        resolved = get_path(applicant, data_path)

    # Use explicit fill_value if data_path didn't resolve or wasn't set
    if resolved is None:
        resolved = fill_value

    # Detect KDDRFI sentinel
    if isinstance(resolved, str) and "KDDRFI" in resolved:
        return None, "pending"

    if resolved is None or resolved == "":
        field_type = field_def.get("field_type", "text")
        if field_type == "checkbox":
            return False, "skip"
        return "", "skip"

    return resolved, "filled"


# ── main logic ────────────────────────────────────────────────────────────────

def build_fill_instructions(schema: dict, applicant: dict) -> list[dict]:
    instructions = []

    for page_key, page_data in schema.get("pages", {}).items():
        if not isinstance(page_data, dict):
            continue
        page_num = page_data.get("pdf_page", page_key)
        section  = page_data.get("form_section", "")

        for field_def in page_data.get("fields", []):
            field_id   = field_def.get("field_id", "")
            field_type = field_def.get("field_type", "text")
            label      = field_def.get("label_human", field_def.get("label_human", ""))
            coords     = field_def.get("coords", {})
            note       = field_def.get("note", "")

            value, status = resolve_value(field_def, applicant)

            # For table_row fields, expand sub-columns into individual ops
            if field_type == "table_row" and isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    instructions.append({
                        "page":       page_num,
                        "section":    section,
                        "field_id":   f"{field_id}.{sub_key}",
                        "label":      f"{label} — {sub_key}",
                        "field_type": "text",
                        "value":      sub_val,
                        "status":     "filled" if sub_val else "skip",
                        "coords":     coords,
                        "note":       note,
                    })
                continue

            instructions.append({
                "page":       page_num,
                "section":    section,
                "field_id":   field_id,
                "label":      label,
                "field_type": field_type,
                "value":      value,
                "status":     status,
                "coords":     coords,
                "note":       note,
            })

    return instructions


def print_report(instructions: list[dict]):
    """Human-readable fill report."""
    total    = len(instructions)
    filled   = sum(1 for i in instructions if i["status"] == "filled")
    pending  = sum(1 for i in instructions if i["status"] == "pending")
    skipped  = sum(1 for i in instructions if i["status"] == "skip")

    print(f"\n{'═'*64}")
    print(f"  FSA-2001 FILL REPORT  —  Scott Weeden")
    print(f"{'═'*64}")
    print(f"  Total fields   : {total}")
    print(f"  ✓ Filled       : {filled}")
    print(f"  ⚠ Pending      : {pending}  (KDDRFI — needs collection)")
    print(f"  ○ Skip/blank   : {skipped}")
    print(f"{'─'*64}")

    if pending:
        print("\n  PENDING FIELDS (must be collected before submission):")
        for item in instructions:
            if item["status"] == "pending":
                print(f"    [{item['page']}] {item['field_id']:40s}  {item['label']}")

    print(f"\n  FILLED FIELDS:")
    for item in instructions:
        if item["status"] == "filled":
            val = str(item["value"])
            if len(val) > 55:
                val = val[:52] + "..."
            print(f"    [{item['page']}] {item['field_id']:40s}  = {val!r}")

    print(f"\n{'═'*64}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="FSA-2001 form-fill instruction generator")
    p.add_argument("--schema", default="fsa2001_field_mapping_schema.json")
    p.add_argument("--data",   default="fsa2001_weeden_data_1.json")
    p.add_argument("--out",    default="fsa2001_fill_instructions.json")
    p.add_argument("--report", action="store_true", help="Print human-readable report")
    args = p.parse_args()

    schema_path = Path(args.schema)
    data_path_  = Path(args.data)

    if not schema_path.exists():
        sys.exit(f"Schema not found: {schema_path}")
    if not data_path_.exists():
        sys.exit(f"Data not found: {data_path_}")

    with open(schema_path) as f:
        schema = json.load(f)
    with open(data_path_) as f:
        applicant = json.load(f)

    instructions = build_fill_instructions(schema, applicant)

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(instructions, f, indent=2)
    print(f"Wrote {len(instructions)} fill instructions → {out_path}")

    if args.report:
        print_report(instructions)


if __name__ == "__main__":
    main()
