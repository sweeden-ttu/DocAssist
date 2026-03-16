#!/usr/bin/env python3
"""
Extract tables with cell coordinates from a PDF using Docling, with support for
two-column layouts (e.g. tax/form with assets left, liabilities right).

Example: FSA-2001 Part H Balance Sheet (page 6) has two side-by-side tables.
Uses table.data.grid (row/col), cell.bbox (.l, .t, .r, .b), and a page midpoint
to classify cells as left (assets) or right (liabilities).

Usage:
  python docling_table_extract.py path/to/form.pdf [--page 6] [--midpoint 300] [--ocr-analysis path/to/ocr.json]
"""

import argparse
import json
from pathlib import Path


def _get_bbox_attrs(bbox):
    """Return left, top, right, bottom from a Docling BoundingBox (same order as .l .t .r .b)."""
    if bbox is None:
        return None
    return getattr(bbox, "l", None), getattr(bbox, "t", None), getattr(bbox, "r", None), getattr(bbox, "b", None)


def extract_tables_two_columns(conv_res, page_midpoint: float = 300):
    """
    Iterate all tables in a ConversionResult; for each cell, get row/col and bbox,
    and classify by x-position into left_column_data (assets) / right_column_data (liabilities).

    Returns list of dicts: [
      {
        "table_ix": int,
        "page_no": int (from table prov if available),
        "left_column_data": [{"text": str, "row": int, "col": int, "coords": {"x", "y", "width", "height"}}, ...],
        "right_column_data": [...],
      },
      ...
    ]
    """
    from docling.datamodel.document import ConversionResult

    if not hasattr(conv_res, "document") or conv_res.document is None:
        return []

    doc = conv_res.document
    if not hasattr(doc, "tables"):
        return []

    out = []
    for table_ix, table in enumerate(doc.tables):
        page_no = None
        if hasattr(table, "prov") and table.prov:
            page_no = getattr(table.prov[0], "page_no", None)

        left_column_data = []
        right_column_data = []

        # table.data.grid is 2D (row, col); fallback to table.data.table_cells (flat)
        grid = getattr(getattr(table, "data", None), "grid", None)
        if grid is not None:
            for row_idx, row in enumerate(grid):
                for col_idx, cell in enumerate(row):
                    bbox_attrs = _get_bbox_attrs(getattr(cell, "bbox", None))
                    if not bbox_attrs or None in bbox_attrs:
                        continue
                    x_left, y_top, x_right, y_bottom = bbox_attrs
                    cell_info = {
                        "text": (getattr(cell, "text", None) or "").strip(),
                        "row": row_idx,
                        "col": col_idx,
                        "coords": {
                            "x": x_left,
                            "y": y_top,
                            "width": x_right - x_left,
                            "height": y_bottom - y_top,
                        },
                    }
                    if x_left < page_midpoint:
                        left_column_data.append(cell_info)
                    else:
                        right_column_data.append(cell_info)
        else:
            table_cells = getattr(getattr(table, "data", None), "table_cells", None) or []
            for idx, cell in enumerate(table_cells):
                bbox_attrs = _get_bbox_attrs(getattr(cell, "bbox", None))
                if not bbox_attrs or None in bbox_attrs:
                    continue
                x_left, y_top, x_right, y_bottom = bbox_attrs
                row_idx = getattr(cell, "start_row_offset_idx", idx)
                col_idx = getattr(cell, "start_col_offset_idx", 0)
                cell_info = {
                    "text": (getattr(cell, "text", None) or "").strip(),
                    "row": row_idx,
                    "col": col_idx,
                    "coords": {
                        "x": x_left,
                        "y": y_top,
                        "width": x_right - x_left,
                        "height": y_bottom - y_top,
                    },
                }
                if x_left < page_midpoint:
                    left_column_data.append(cell_info)
                else:
                    right_column_data.append(cell_info)

        out.append({
            "table_ix": table_ix,
            "page_no": page_no,
            "left_column_data": left_column_data,
            "right_column_data": right_column_data,
        })

    return out


def map_cells_to_form_fields(table_extracts, ocr_analysis_path: Path, page_height_pt: float = 792.0):
    """
    Map extracted table cells to OCR analysis form fields by coordinate overlap.
    OCR field_coords use PDF convention: origin bottom-left, (x, y, width, height).
    Docling bbox is often top-left origin; we treat cell coords as same as page (points).
    Returns list of assignments: { "page_num", "field_index", "field_id", "answer", "cell_text" }.
    """
    with open(ocr_analysis_path, encoding="utf-8") as f:
        ocr = json.load(f)

    assignments = []
    for block in table_extracts:
        page_num = block.get("page_no")
        if page_num is None:
            continue
        page_key = f"page_{page_num}"
        page_data = ocr.get(page_key)
        if not page_data:
            continue
        fields = page_data.get("fields", [])

        def pdf_bbox_to_rect(coords):
            # OCR: x,y = bottom-left of field; width, height
            x = coords.get("x", 0)
            y = coords.get("y", 0)
            w = coords.get("width", 0)
            h = coords.get("height", 0)
            # PDF: y is from bottom; rect left, bottom, right, top
            left = x
            right = x + w
            bottom = y
            top = y + h
            return left, bottom, right, top

        for side in ("left_column_data", "right_column_data"):
            for cell in block.get(side, []):
                cx = cell["coords"]["x"]
                cy = cell["coords"]["y"]
                cw = cell["coords"]["width"]
                ch = cell["coords"]["height"]
                # Assume Docling bbox: top-left origin (y down). Convert to PDF-like rect for overlap.
                cell_left = cx
                cell_right = cx + cw
                cell_top = cy
                cell_bottom = cy + ch

                for field_idx, field in enumerate(fields):
                    fc = field.get("field_coords") or {}
                    if fc.get("page") != page_num:
                        continue
                    fl, fb, fr, ft = pdf_bbox_to_rect(fc)
                    # Overlap in x and y (PDF: y up, so field top > field bottom)
                    overlap_x = not (cell_right < fl or cell_left > fr)
                    overlap_y = not (cell_bottom > ft or cell_top < fb)
                    if overlap_x and overlap_y and cell.get("text"):
                        assignments.append({
                            "page_num": page_num,
                            "field_index": field_idx,
                            "field_id": field.get("field", ""),
                            "answer": cell["text"],
                            "cell_text": cell["text"],
                        })
                        break

    return assignments


def main():
    ap = argparse.ArgumentParser(description="Extract tables with cell coordinates (two-column layout).")
    ap.add_argument("pdf", type=Path, help="Path to PDF")
    ap.add_argument("--page", type=int, default=None, help="Only convert this page (1-based); omit for all.")
    ap.add_argument("--midpoint", type=float, default=300, help="X threshold for left vs right column (default 300).")
    ap.add_argument("--ocr-analysis", type=Path, default=None, help="OCR analysis JSON to map cells to form fields.")
    ap.add_argument("--out", type=Path, default=None, help="Write table extraction JSON here.")
    args = ap.parse_args()

    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import PdfFormatOption
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_page_images = True
    doc_converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=pipeline_options,
            )
        },
    )

    pdf_path = args.pdf.resolve()
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    kwargs = {}
    if args.page is not None:
        kwargs["page_range"] = (args.page, args.page)

    print("Converting PDF with Docling...")
    conv_res = doc_converter.convert(str(pdf_path), **kwargs)

    table_extracts = extract_tables_two_columns(conv_res, page_midpoint=args.midpoint)
    print(f"Extracted {len(table_extracts)} table(s).")

    for i, block in enumerate(table_extracts):
        print(f"  Table {i} (page_no={block.get('page_no')}): left={len(block['left_column_data'])} cells, right={len(block['right_column_data'])} cells")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(table_extracts, f, indent=2)
        print(f"Wrote {args.out}")

    if args.ocr_analysis and args.ocr_analysis.exists():
        assignments = map_cells_to_form_fields(table_extracts, args.ocr_analysis)
        print(f"Mapped {len(assignments)} cell(s) to form fields.")
        for a in assignments:
            print(f"  page {a['page_num']} field {a['field_index']} ({a['field_id']}): {a['answer']!r}")


if __name__ == "__main__":
    main()
