#!/usr/bin/env python3
"""
FSA-2001 pipeline: OCR analysis → VLM label fix → fill-from-schema → overlays with values.

Steps:
  1. ocr_field_analysis.py     → writes {stem}_ocr_analysis.json
  2. post_process_ocr_with_vlm.py --fix-labels → updates same JSON with better labels
  3. fsa2001_fill_from_schema.py  → writes fsa2001_fill_instructions.json
  4. generate_multi_analysis_overlays.py → overlays + annotated PNGs with bounding boxes
     and values from fsa2001_field_mapping_schema.json + fsa2001_fill_instructions.json

Usage:
  python run_fsa2001_pipeline.py [--pdf PATH] [--data PATH] [--skip-ocr] [--skip-vlm] [--skip-fill] [--skip-overlays]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Run FSA-2001 OCR → VLM → fill → overlays pipeline")
    ap.add_argument("--pdf", type=Path, default=None, help="PDF path (default: templates/usda/FSA2001_250321V05LC (14).pdf)")
    ap.add_argument("--data", type=Path, default=None, help="Applicant data JSON for fill-from-schema (default: annotations/usda/schemas/fsa2001_weeden_data_1.json)")
    ap.add_argument("--annotations-dir", type=Path, default=None, help="Annotations output dir (default: annotations/usda)")
    ap.add_argument("--schemas-dir", type=Path, default=None, help="Schemas dir (default: annotations/usda/schemas)")
    ap.add_argument("--skip-ocr", action="store_true", help="Skip step 1 (OCR field analysis)")
    ap.add_argument("--skip-vlm", action="store_true", help="Skip step 2 (VLM label fix)")
    ap.add_argument("--skip-fill", action="store_true", help="Skip step 3 (fill from schema)")
    ap.add_argument("--skip-overlays", action="store_true", help="Skip step 4 (overlays)")
    ap.add_argument("--report", action="store_true", help="Print fill report in step 3")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent
    annotations_dir = args.annotations_dir or (project_root / "annotations" / "usda")
    schemas_dir = args.schemas_dir or (annotations_dir / "schemas")
    pdf_path = args.pdf or (project_root / "templates" / "usda" / "FSA2001_250321V05LC (14).pdf")
    data_path = args.data or (schemas_dir / "fsa2001_weeden_data_1.json")

    if not pdf_path.exists():
        sys.exit(f"PDF not found: {pdf_path}")

    stem = pdf_path.stem
    ocr_analysis_path = annotations_dir / f"{stem}_ocr_analysis.json"
    field_mapping_schema_path = schemas_dir / "fsa2001_field_mapping_schema.json"
    fill_instructions_path = schemas_dir / "fsa2001_fill_instructions.json"

    # Step 1: OCR field analysis
    if not args.skip_ocr:
        print("\n" + "=" * 60 + "\n  STEP 1: OCR field analysis\n" + "=" * 60)
        from ocr_field_analysis import analyze_pdf_with_ocr
        analyze_pdf_with_ocr(str(pdf_path), str(annotations_dir))
        if not ocr_analysis_path.exists():
            sys.exit(f"Expected output not found: {ocr_analysis_path}")
    else:
        print("\n[SKIP] Step 1: OCR field analysis")
        if not ocr_analysis_path.exists():
            sys.exit(f"Cannot skip OCR: {ocr_analysis_path} not found")

    # Step 2: VLM fix labels
    if not args.skip_vlm:
        print("\n" + "=" * 60 + "\n  STEP 2: VLM fix labels\n" + "=" * 60)
        from post_process_ocr_with_vlm import fix_ocr_analysis_labels
        fix_ocr_analysis_labels(pdf_path, ocr_analysis_path)
    else:
        print("\n[SKIP] Step 2: VLM fix labels")

    # Step 3: Fill from schema
    if not args.skip_fill:
        print("\n" + "=" * 60 + "\n  STEP 3: Fill from schema\n" + "=" * 60)
        if not field_mapping_schema_path.exists():
            sys.exit(f"Schema not found: {field_mapping_schema_path}")
        if not data_path.exists():
            print(f"Applicant data not found: {data_path}. Skipping step 3 (use --data PATH to provide it).")
            args.skip_fill = True
        else:
            cmd = [
                sys.executable,
                str(project_root / "fsa2001_fill_from_schema.py"),
                "--schema", str(field_mapping_schema_path),
                "--data", str(data_path),
                "--out", str(fill_instructions_path),
            ]
            if args.report:
                cmd.append("--report")
            r = subprocess.run(cmd, cwd=str(project_root))
            if r.returncode != 0:
                sys.exit(f"Step 3 failed with exit code {r.returncode}")
            if not fill_instructions_path.exists():
                sys.exit(f"Expected output not found: {fill_instructions_path}")
    else:
        print("\n[SKIP] Step 3: Fill from schema")
        if not fill_instructions_path.exists():
            print(f"Warning: {fill_instructions_path} not found; step 4 will omit fill values.")

    # Step 4: Overlays with bounding boxes and values
    if not args.skip_overlays:
        print("\n" + "=" * 60 + "\n  STEP 4: Overlays (boxes + values)\n" + "=" * 60)
        from generate_multi_analysis_overlays import run_overlays_with_fill_data
        run_overlays_with_fill_data(
            pdf_path=pdf_path,
            ocr_analysis_path=ocr_analysis_path,
            overlays_dir=annotations_dir / "overlays",
            annotated_dir=annotations_dir / "annotated",
            fill_instructions_path=fill_instructions_path if fill_instructions_path.exists() else None,
            field_mapping_schema_path=field_mapping_schema_path if field_mapping_schema_path.exists() else None,
        )
    else:
        print("\n[SKIP] Step 4: Overlays")

    print("\n" + "=" * 60 + "\n  Pipeline complete\n" + "=" * 60)


if __name__ == "__main__":
    main()
