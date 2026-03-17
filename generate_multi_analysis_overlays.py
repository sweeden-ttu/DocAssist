#!/usr/bin/env python3
"""
Generate transparent multi-analysis overlays and annotated page images.

For each analysis JSON, this script:
- Extracts bounding boxes for each page.
- Draws them on a transparent RGBA overlay image, using a distinct color
  per analysis source.
- Renders the corresponding PDF page and composites the overlay on top.

Outputs:
- Overlays: `annotations/usda/overlays/page_{n}_multi_overlay.png`
  (transparent, colored boxes only)
- Annotated pages: `annotations/usda/annotated/page_{n}_multi_annotated.png`
  (PDF page with overlay composited)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont


@dataclass
class AnalysisConfig:
    name: str
    path: Path
    color: Tuple[int, int, int, int]  # RGBA


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def extract_boxes_from_coordinates(data: dict) -> Dict[int, List[Tuple[float, float, float, float]]]:
    """Extract [x0, y0, x1, y1] boxes from *_coordinates.json."""
    boxes_by_page: Dict[int, List[Tuple[float, float, float, float]]] = {}

    form_fields = data.get("form_fields", {})
    for field_info in form_fields.values():
        page = int(field_info.get("page", 1))
        bbox = field_info.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(v, (int, float)) for v in bbox)
        ):
            continue

        x0, y0, x1, y1 = map(float, bbox)
        boxes_by_page.setdefault(page, []).append((x0, y0, x1, y1))

    return boxes_by_page


def extract_boxes_from_page_fields(data: dict) -> Dict[int, List[Tuple[float, float, float, float]]]:
    """
    Extract [x0, y0, x1, y1] boxes from analyses that are shaped as:
    {
      "page_3": {
        "fields": [
          {
            "field_coords": {
              "page": 3,
              "x": ...,
              "y": ...,
              "width": ...,
              "height": ...
            }
          },
          ...
        ]
      },
      ...
    }
    """
    boxes_by_page: Dict[int, List[Tuple[float, float, float, float]]] = {}

    for key, page_info in data.items():
        if not key.startswith("page_"):
            continue

        try:
            page_num = int(key.split("_")[1])
        except (IndexError, ValueError):
            continue

        fields: Iterable[dict] = page_info.get("fields", [])
        for field in fields:
            coords = field.get("field_coords") or {}
            x = coords.get("x")
            y = coords.get("y")
            w = coords.get("width")
            h = coords.get("height")
            if not all(isinstance(v, (int, float)) for v in (x, y, w, h)):
                continue

            x0 = float(x)
            y0 = float(y)
            x1 = x0 + float(w)
            y1 = y0 + float(h)

            boxes_by_page.setdefault(page_num, []).append((x0, y0, x1, y1))

    return boxes_by_page


def _cell_coords_to_pdf_box(
    coords: dict,
    page_height_pt: float,
) -> Optional[Tuple[float, float, float, float]]:
    """Convert cell coords (x, y, width, height, top-left origin) to PDF (x0, y0, x1, y1)."""
    x = coords.get("x")
    y = coords.get("y")
    w = coords.get("width")
    h = coords.get("height")
    if None in (x, y, w, h):
        return None
    x, y, w, h = float(x), float(y), float(w), float(h)
    x0 = x
    y0 = page_height_pt - y - h
    x1 = x + w
    y1 = page_height_pt - y
    return (x0, y0, x1, y1)


def extract_boxes_from_docling_tables(
    data: list,
    page_height_pt: float = 792.0,
) -> Dict[int, List[Tuple[float, float, float, float]]]:
    """
    Extract [x0, y0, x1, y1] boxes from Docling table JSON (list of blocks with
    page_no, left_column_data, right_column_data). Each cell has coords in top-left
    origin; convert to PDF coords (bottom-left) for overlay.
    """
    boxes_by_page: Dict[int, List[Tuple[float, float, float, float]]] = {}
    if not isinstance(data, list):
        return boxes_by_page
    for block in data:
        page_no = block.get("page_no")
        if page_no is None:
            continue
        for side in ("left_column_data", "right_column_data"):
            for cell in block.get(side, []):
                coords = cell.get("coords") or {}
                box = _cell_coords_to_pdf_box(coords, page_height_pt)
                if box:
                    boxes_by_page.setdefault(page_no, []).append(box)
    return boxes_by_page


def extract_boxes_from_page6_tables(
    data: dict,
    page_height_pt: float = 792.0,
) -> Dict[int, List[Tuple[float, float, float, float]]]:
    """
    Extract [x0, y0, x1, y1] from single-page table JSON with "cells" and "page"
    (e.g. page6_tables.json). Each cell has coords: x, y, width, height (top-left).
    """
    boxes_by_page: Dict[int, List[Tuple[float, float, float, float]]] = {}
    if not isinstance(data, dict):
        return boxes_by_page
    page_no = data.get("page")
    cells = data.get("cells")
    if page_no is None or not isinstance(cells, list):
        return boxes_by_page
    page_no = int(page_no)
    for cell in cells:
        coords = cell.get("coords") or {}
        box = _cell_coords_to_pdf_box(coords, page_height_pt)
        if box:
            boxes_by_page.setdefault(page_no, []).append(box)
    return boxes_by_page


def get_page6_table_cells_with_answers(
    path: Path,
    page_height_pt: float = 792.0,
) -> Dict[int, List[Tuple[Tuple[float, float, float, float], str]]]:
    """
    Load page6_tables.json (single object with "page" and "cells") and return
    page_num -> [(box_pdf, answer_text), ...]. Uses cell "answer" if non-empty, else "text".
    """
    result: Dict[int, List[Tuple[Tuple[float, float, float, float], str]]] = {}
    data = load_json(path)
    if not isinstance(data, dict) or "cells" not in data or "page" not in data:
        return result
    page_no = int(data["page"])
    cells = data.get("cells", [])
    for cell in cells:
        coords = cell.get("coords") or {}
        box = _cell_coords_to_pdf_box(coords, page_height_pt)
        if not box:
            continue
        answer = (cell.get("answer") or "").strip() or (cell.get("text") or "").strip()
        result.setdefault(page_no, []).append((box, answer))
    return result


def get_page6_inferred_cells(
    path: Path,
    page_height_pt: float = 792.0,
) -> List[Tuple[Tuple[float, float, float, float], str]]:
    """
    Load page6_tables.json and return only cells where coords.inferred is True.
    Returns [(box_pdf, answer_text), ...] for overlay (no magenta boxes).
    """
    data = load_json(path)
    if not isinstance(data, dict) or "cells" not in data or "page" not in data:
        return []
    cells = data.get("cells", [])
    out: List[Tuple[Tuple[float, float, float, float], str]] = []
    for cell in cells:
        coords = cell.get("coords") or {}
        if not coords.get("inferred"):
            continue
        box = _cell_coords_to_pdf_box(coords, page_height_pt)
        if not box:
            continue
        answer = (cell.get("answer") or "").strip() or (cell.get("text") or "").strip()
        if not answer:
            continue
        out.append((box, answer))
    return out


def get_page7_value_cells(
    path: Path,
    page_height_pt: float = 792.0,
) -> List[Tuple[Tuple[float, float, float, float], str]]:
    """
    Load page7_tables.json and return only cells where cell_type is "value".
    Returns [(box_pdf, answer_text), ...] for overlay (no magenta boxes).
    """
    data = load_json(path)
    if not isinstance(data, dict) or "cells" not in data or "page" not in data:
        return []
    cells = data.get("cells", [])
    out: List[Tuple[Tuple[float, float, float, float], str]] = []
    for cell in cells:
        if cell.get("cell_type") != "value":
            continue
        coords = cell.get("coords") or {}
        box = _cell_coords_to_pdf_box(coords, page_height_pt)
        if not box:
            continue
        answer = (cell.get("answer") or "").strip() or (cell.get("text") or "").strip()
        out.append((box, answer))
    return out


def extract_boxes_generic(path: Path) -> Dict[int, List[Tuple[float, float, float, float]]]:
    """Dispatch extraction based on the known file pattern."""
    data = load_json(path)

    if "form_fields" in data:
        # Coordinates-style data
        return extract_boxes_from_coordinates(data)

    # Docling tables JSON: list of { page_no, left_column_data, right_column_data }
    if isinstance(data, list) and data and isinstance(data[0], dict):
        first = data[0]
        if "left_column_data" in first or "right_column_data" in first:
            return extract_boxes_from_docling_tables(data)

    # Single-page table JSON: { "page": 6, "cells": [ { "coords": {...} }, ... ] }
    if isinstance(data, dict) and "cells" in data and "page" in data:
        return extract_boxes_from_page6_tables(data)

    # Default: per-page `fields` structure
    return extract_boxes_from_page_fields(data)


def get_ocr_fields_by_page(ocr_analysis_data: dict) -> Dict[int, List[dict]]:
    """
    From OCR analysis JSON (page_N.fields with field_coords + answer), return
    page_num -> list of field dicts with 'field_coords' and 'answer'.
    """
    by_page: Dict[int, List[dict]] = {}
    for key, page_info in ocr_analysis_data.items():
        if not key.startswith("page_"):
            continue
        try:
            page_num = int(key.split("_")[1])
        except (IndexError, ValueError):
            continue
        for field in page_info.get("fields", []):
            coords = field.get("field_coords") or {}
            if coords.get("x") is None:
                continue
            by_page.setdefault(page_num, []).append({
                "field_coords": coords,
                "answer": field.get("answer"),
            })
    return by_page


def _get_bold_dark_blue_font(size: int = 10) -> ImageFont.FreeTypeFont:
    """Return a bold font for answer text; dark blue is applied at draw time."""
    bold_paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in bold_paths:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def build_all_boxes(
    analyses: Iterable[AnalysisConfig],
) -> Dict[str, Dict[int, List[Tuple[float, float, float, float]]]]:
    """
    Return mapping:
      analysis_name -> { page_num -> [ (x0, y0, x1, y1), ... ] }
    """
    all_boxes: Dict[str, Dict[int, List[Tuple[float, float, float, float]]]] = {}

    for analysis in analyses:
        boxes_by_page = extract_boxes_generic(analysis.path)
        all_boxes[analysis.name] = boxes_by_page

    return all_boxes


# Bold dark blue for answer text inside boxes (RGB)
ANSWER_TEXT_COLOR = (0, 51, 139, 255)  # dark blue


def _split_text_two_lines(text: str, max_chars_per_line: int) -> Tuple[str, str]:
    """Split text into two lines at a word boundary near the middle."""
    text = text.strip()
    if len(text) <= max_chars_per_line:
        return (text, "")
    mid = len(text) // 2
    # Prefer split at space
    for i in range(mid, min(mid + 20, len(text))):
        if i < len(text) and text[i] == " ":
            return (text[:i].strip(), text[i:].strip())
    for i in range(mid, max(0, mid - 20), -1):
        if i > 0 and text[i - 1] == " ":
            return (text[: i - 1].strip(), text[i:].strip())
    return (text[:mid].strip(), text[mid:].strip())


def _draw_answer_in_box(
    draw: ImageDraw.ImageDraw,
    left: float,
    top: float,
    right: float,
    bottom: float,
    answer: str,
    font: ImageFont.FreeTypeFont,
    page_num: Optional[int] = None,
    page_5_font: Optional[ImageFont.FreeTypeFont] = None,
) -> None:
    """Draw answer inside a bounding box in bold dark blue. For Yes/checkbox use centered X. On page 5 use smaller font and wrap in two lines."""
    if answer is None:
        return
    if isinstance(answer, bool):
        mark = "X" if answer else ""
    else:
        mark = str(answer).strip()
    if not mark:
        return
    box_w = max(1, right - left)
    box_h = max(1, bottom - top)
    # Use smaller font for page 5
    use_font = font
    wrap_two_lines = False
    if page_num == 5 and page_5_font is not None:
        use_font = page_5_font
        wrap_two_lines = True
    # Checkbox/control: "Yes", "yes", True → draw single "X" centered on control
    if (
        isinstance(answer, bool) and answer
        or (isinstance(mark, str) and mark.upper() == "YES")
    ):
        try:
            use = "X"
            bbox = draw.textbbox((0, 0), use, font=use_font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = left + (box_w - tw) / 2
            ty = top + (box_h - th) / 2
            draw.text((tx, ty), use, fill=ANSWER_TEXT_COLOR, font=use_font)
        except Exception:
            try:
                draw.text((left + box_w / 2 - 4, top + box_h / 2 - 6), "X", fill=ANSWER_TEXT_COLOR, font=use_font)
            except Exception:
                pass
        return
    # Text field: on page 5 wrap in two lines; else truncate to fit
    max_chars = max(1, int(box_w / 5))
    if wrap_two_lines and len(mark) > max_chars:
        line1, line2 = _split_text_two_lines(mark, max_chars)
        try:
            try:
                bbox = draw.textbbox((0, 0), "Ay", font=use_font)
                line_height = bbox[3] - bbox[1] + 1
            except Exception:
                line_height = 12
            tx, ty = left + 2, top + 1
            draw.text((tx, ty), line1, fill=ANSWER_TEXT_COLOR, font=use_font)
            if line2:
                draw.text((tx, ty + line_height), line2, fill=ANSWER_TEXT_COLOR, font=use_font)
        except Exception:
            if len(mark) > max_chars:
                mark = mark[: max_chars - 3] + "..."
            draw.text((left + 2, top + 1), mark, fill=ANSWER_TEXT_COLOR, font=use_font)
    else:
        if len(mark) > max_chars:
            mark = mark[: max_chars - 3] + "..."
        try:
            tx, ty = left + 2, top + 1
            draw.text((tx, ty), mark, fill=ANSWER_TEXT_COLOR, font=use_font)
        except Exception:
            pass


def create_overlay_for_page(
    page,
    page_num: int,
    page_height: float,
    analyses: Iterable[AnalysisConfig],
    boxes_by_analysis: Dict[str, Dict[int, List[Tuple[float, float, float, float]]]],
    scale: float = 2.0,
    ocr_fields_for_page: Optional[List[dict]] = None,
    answer_font: Optional[ImageFont.FreeTypeFont] = None,
    page_5_font: Optional[ImageFont.FreeTypeFont] = None,
    table_cells_for_page: Optional[List[Tuple[Tuple[float, float, float, float], str]]] = None,
) -> Image.Image:
    """Create a transparent overlay image for a given page. Optionally draw answer text inside OCR boxes and inside docling_tables (page6_tables) cells. Page 5 uses page_5_font and wraps long answers in two lines."""
    width = int(page.get_width() * scale)
    height = int(page.get_height() * scale)

    overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    for analysis in analyses:
        page_boxes = boxes_by_analysis.get(analysis.name, {}).get(page_num, [])
        if not page_boxes:
            continue

        use_answers = (
            analysis.name == "ocr"
            and ocr_fields_for_page is not None
            and len(ocr_fields_for_page) == len(page_boxes)
            and answer_font is not None
        )

        for idx, (x0, y0, x1, y1) in enumerate(page_boxes):
            # Coordinates are in PDF space with origin at bottom-left.
            # Convert to image coordinates (origin at top-left).
            x0_s = x0 * scale
            x1_s = x1 * scale

            # y0/y1 measured from bottom; convert to top/bottom in image space.
            y0_img = (page_height - y0) * scale
            y1_img = (page_height - y1) * scale

            top = min(y0_img, y1_img)
            bottom = max(y0_img, y1_img)

            # Do not draw magenta bounding box for OCR; only draw answer text. Skip boxes for docling_tables on page 6.
            if analysis.name != "ocr" and not (analysis.name == "docling_tables" and page_num == 6):
                draw.rectangle(
                    [x0_s, top, x1_s, bottom],
                    outline=analysis.color,
                    width=2,
                )

            if use_answers:
                answer = ocr_fields_for_page[idx].get("answer")
                _draw_answer_in_box(
                    draw, x0_s, top, x1_s, bottom, answer or "",
                    answer_font, page_num=page_num, page_5_font=page_5_font,
                )

            # Draw answer text inside docling_tables (page6_tables) cells
            if (
                analysis.name == "docling_tables"
                and table_cells_for_page is not None
                and idx < len(table_cells_for_page)
                and answer_font is not None
            ):
                _, cell_answer = table_cells_for_page[idx]
                if cell_answer:
                    _draw_answer_in_box(
                        draw, x0_s, top, x1_s, bottom, cell_answer,
                        answer_font, page_num=page_num, page_5_font=page_5_font,
                    )

    return overlay


def generate_multi_analysis_overlays(
    pdf_path: Path,
    analyses: Iterable[AnalysisConfig],
    overlays_dir: Path,
    annotated_dir: Path,
    scale: float = 2.0,
    ocr_analysis_path: Optional[Path] = None,
) -> None:
    """
    Generate overlay and annotated PNGs per page. If ocr_analysis_path is set (or
    an analysis named 'ocr' is present), loads that JSON and draws the 'answer'
    field inside each bounding box in bold dark blue.
    """
    overlays_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    pdf = pdfium.PdfDocument(str(pdf_path))
    num_pages = len(pdf)

    boxes_by_analysis = build_all_boxes(analyses)

    # Load OCR analysis for answer text inside boxes (bold dark blue)
    ocr_fields_by_page: Dict[int, List[dict]] = {}
    ocr_path = ocr_analysis_path
    if ocr_path is None:
        for a in analyses:
            if a.name == "ocr":
                ocr_path = a.path
                break
    if ocr_path and ocr_path.exists():
        ocr_data = load_json(ocr_path)
        ocr_fields_by_page = get_ocr_fields_by_page(ocr_data)
    answer_font = _get_bold_dark_blue_font(size=20)
    page_5_font = _get_bold_dark_blue_font(size=10)  # smaller on page 5 with wrap

    # Assume all analyses share the same page height as the PDF.
    first_page = pdf.get_page(0)
    page_height = float(first_page.get_height())

    # Load page6_tables cell answers for docling_tables analysis (transparent overlay with answers)
    table_cells_by_page: Dict[int, List[Tuple[Tuple[float, float, float, float], str]]] = {}
    page6_tables_path: Optional[Path] = None
    for a in analyses:
        if a.name == "docling_tables" and a.path.exists():
            page6_tables_path = a.path
            data = load_json(a.path)
            if isinstance(data, dict) and "cells" in data and "page" in data:
                table_cells_by_page.update(get_page6_table_cells_with_answers(a.path, page_height))
            break

    # Page 6 docling_tables: use only inferred boxes so overlay indices match inferred-only cell list.
    if page6_tables_path and page6_tables_path.exists():
        inferred_cells = get_page6_inferred_cells(page6_tables_path, page_height)
        if inferred_cells and "docling_tables" in boxes_by_analysis and 6 in boxes_by_analysis["docling_tables"]:
            boxes_by_analysis["docling_tables"][6] = [box for box, _ in inferred_cells]

    # Page 6 only: use page_6.png as base, overlay only inferred cells (no magenta boxes).
    annotations_dir = overlays_dir.parent
    page_6_base = annotations_dir / "page_6.png"
    page_7_base = annotations_dir / "page_7.png"
    page7_tables_path = annotations_dir / "page7_tables.json"
    if page6_tables_path and page6_tables_path.exists() and page_6_base.exists():
        inferred_cells = get_page6_inferred_cells(page6_tables_path, page_height)
        if inferred_cells:
            base_img = Image.open(page_6_base).convert("RGBA")
            w, h = base_img.size
            page_width_pt = 612.0
            scale_x = w / page_width_pt
            scale_y = h / page_height
            overlay = Image.new("RGBA", (w, h), (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)
            font = _get_bold_dark_blue_font(size=20)
            for (x0, y0, x1, y1), answer in inferred_cells:
                left = x0 * scale_x
                right = x1 * scale_x
                top = (page_height - y1) * scale_y
                bottom = (page_height - y0) * scale_y
                _draw_answer_in_box(draw, left, top, right, bottom, answer, font)
            overlay_path = overlays_dir / "page_6_multi_overlay.png"
            overlay.save(overlay_path, "PNG")
            combined = Image.alpha_composite(base_img, overlay)
            annotated_path = annotated_dir / "page_6_multi_annotated.png"
            combined.save(annotated_path, "PNG")
            print("Page 6 (from page_6.png + inferred cells only):")
            print(f"  Overlay   -> {overlay_path}")
            print(f"  Annotated -> {annotated_path}")

    # Page 7 only: use page_7.png as base, overlay only cell_type "value" cells (no magenta boxes).
    if page7_tables_path.exists() and page_7_base.exists():
        value_cells = get_page7_value_cells(page7_tables_path, page_height)
        if value_cells:
            base_img = Image.open(page_7_base).convert("RGBA")
            w, h = base_img.size
            page_width_pt = 612.0
            scale_x = w / page_width_pt
            scale_y = h / page_height
            overlay = Image.new("RGBA", (w, h), (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)
            font = _get_bold_dark_blue_font(size=20)
            for (x0, y0, x1, y1), answer in value_cells:
                left = x0 * scale_x
                right = x1 * scale_x
                top = (page_height - y1) * scale_y
                bottom = (page_height - y0) * scale_y
                _draw_answer_in_box(draw, left, top, right, bottom, answer, font)
            overlay_path = overlays_dir / "page_7_multi_overlay.png"
            overlay.save(overlay_path, "PNG")
            combined = Image.alpha_composite(base_img, overlay)
            annotated_path = annotated_dir / "page_7_multi_annotated.png"
            combined.save(annotated_path, "PNG")
            print("Page 7 (from page_7.png + value cells only):")
            print(f"  Overlay   -> {overlay_path}")
            print(f"  Annotated -> {annotated_path}")

    # Pages 8, 9, 10: same logic as page 7 — PNG base + pageN_tables.json, only cell_type "value", draw "answer" at (x,y).
    for page_num in (8, 9, 10):
        page_base = annotations_dir / f"page_{page_num}.png"
        page_tables_path = annotations_dir / f"page{page_num}_tables.json"
        if not page_tables_path.exists() or not page_base.exists():
            continue
        base_img = Image.open(page_base).convert("RGBA")
        w, h = base_img.size
        # Table coords are in same space as PNG (e.g. Docling export), so use image size as logical page size.
        page_height_pt = float(h)
        page_width_pt = float(w)
        value_cells = get_page7_value_cells(page_tables_path, page_height_pt)
        if not value_cells:
            continue
        scale_x = w / page_width_pt
        scale_y = h / page_height_pt
        overlay = Image.new("RGBA", (w, h), (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        font = _get_bold_dark_blue_font(size=20)
        for (x0, y0, x1, y1), answer in value_cells:
            left = x0 * scale_x
            right = x1 * scale_x
            top = (page_height_pt - y1) * scale_y
            bottom = (page_height_pt - y0) * scale_y
            _draw_answer_in_box(draw, left, top, right, bottom, answer, font)
        overlay_path = overlays_dir / f"page_{page_num}_multi_overlay.png"
        overlay.save(overlay_path, "PNG")
        combined = Image.alpha_composite(base_img, overlay)
        annotated_path = annotated_dir / f"page_{page_num}_multi_annotated.png"
        combined.save(annotated_path, "PNG")
        print(f"Page {page_num} (from page_{page_num}.png + value cells only):")
        print(f"  Overlay   -> {overlay_path}")
        print(f"  Annotated -> {annotated_path}")

    for page_index in range(num_pages):
        page_num = page_index + 1
        if page_num == 6 and page6_tables_path and page_6_base.exists():
            continue
        if page_num == 7 and page7_tables_path.exists() and page_7_base.exists():
            continue
        if page_num in (8, 9, 10):
            page_base = annotations_dir / f"page_{page_num}.png"
            page_tables_path = annotations_dir / f"page{page_num}_tables.json"
            if page_tables_path.exists() and page_base.exists():
                continue
        page = pdf.get_page(page_index)

        ocr_fields_for_page = ocr_fields_by_page.get(page_num)
        table_cells_for_page = table_cells_by_page.get(page_num)
        if page_num == 6 and page6_tables_path and page6_tables_path.exists():
            table_cells_for_page = get_page6_inferred_cells(page6_tables_path, page_height) or None

        # Create overlay for this page across all analyses; draw answers inside OCR and table cells.
        overlay = create_overlay_for_page(
            page=page,
            page_num=page_num,
            page_height=page_height,
            analyses=analyses,
            boxes_by_analysis=boxes_by_analysis,
            scale=scale,
            ocr_fields_for_page=ocr_fields_for_page,
            answer_font=answer_font,
            page_5_font=page_5_font,
            table_cells_for_page=table_cells_for_page,
        )

        overlay_path = overlays_dir / f"page_{page_num}_multi_overlay.png"
        overlay.save(overlay_path, "PNG")

        # Render base page and composite overlay to create annotated image.
        bitmap = page.render(scale=scale)
        base_img = bitmap.to_pil().convert("RGBA")

        combined = Image.alpha_composite(base_img, overlay)
        annotated_path = annotated_dir / f"page_{page_num}_multi_annotated.png"
        combined.save(annotated_path, "PNG")

        print(f"Page {page_num}:")
        print(f"  Overlay   -> {overlay_path}")
        print(f"  Annotated -> {annotated_path}")


def _group_fill_instructions_by_page(instructions: List[dict]) -> Dict[int, List[dict]]:
    """Group fill_instructions list by page number."""
    by_page: Dict[int, List[dict]] = {}
    for item in instructions:
        p = item.get("page")
        if p is not None:
            by_page.setdefault(int(p), []).append(item)
    return by_page


def _field_mapping_fields_by_page(schema: dict) -> Dict[int, List[dict]]:
    """Flatten field_mapping_schema pages into page -> list of field dicts with coords."""
    by_page: Dict[int, List[dict]] = {}
    for page_key, page_data in schema.get("pages", {}).items():
        if not isinstance(page_data, dict):
            continue
        p = page_data.get("pdf_page")
        if p is None and page_key.startswith("page_"):
            try:
                p = int(page_key.split("_")[1])
            except (IndexError, ValueError):
                continue
        if p is None:
            continue
        for f in page_data.get("fields", []):
            by_page.setdefault(int(p), []).append(f)
    return by_page


def _draw_fill_data_on_overlay(
    draw: ImageDraw.ImageDraw,
    page_num: int,
    page_height_pt: float,
    scale: float,
    fill_instructions: Optional[List[dict]],
    field_mapping_fields: Optional[List[dict]],
    box_color: Tuple[int, int, int, int],
    text_color: Tuple[int, int, int, int],
    font: Optional[ImageFont.FreeTypeFont],
) -> None:
    """Draw bounding boxes and label/value text from fill_instructions and field_mapping onto draw."""
    def pdf_to_img_rect(x: float, y: float, w: float, h: float):
        # PDF: (x, y) = bottom-left of rect, w, h = width, height
        left = x * scale
        right = (x + w) * scale
        top = (page_height_pt - (y + h)) * scale
        bottom = (page_height_pt - y) * scale
        return (left, top, right, bottom)

    # Prefer fill_instructions (has resolved value); fallback to field_mapping for label/fill_value
    items: List[dict] = []
    if fill_instructions:
        for it in fill_instructions:
            c = it.get("coords") or {}
            if not c or c.get("x") is None:
                continue
            items.append({
                "coords": c,
                "label": it.get("label", ""),
                "value": it.get("value"),
                "field_id": it.get("field_id", ""),
            })
    if field_mapping_fields and not items:
        for f in field_mapping_fields:
            c = f.get("coords") or {}
            if not c or c.get("x") is None:
                continue
            items.append({
                "coords": c,
                "label": f.get("label_human", f.get("label", "")),
                "value": f.get("fill_value"),
                "field_id": f.get("field_id", ""),
            })

    for it in items:
        c = it["coords"]
        x, y, w, h = c.get("x", 0), c.get("y", 0), c.get("w", 0), c.get("h", 0)
        if w == 0 and "width" in c:
            w = c["width"]
        if h == 0 and "height" in c:
            h = c["height"]
        left, top, right, bottom = pdf_to_img_rect(float(x), float(y), float(w), float(h))
        draw.rectangle([left, top, right, bottom], outline=box_color, width=2)
        label = (it.get("label") or "")[:50]
        val = it.get("value")
        val_str = "" if val is None else str(val)
        if len(val_str) > 40:
            val_str = val_str[:37] + "..."
        text_line = f"{label} = {val_str}" if val_str else label
        if text_line and font:
            try:
                draw.text((left, max(0, top - 12)), text_line[:60], fill=text_color, font=font)
            except Exception:
                pass


def run_overlays_with_fill_data(
    pdf_path: Path,
    ocr_analysis_path: Path,
    overlays_dir: Path,
    annotated_dir: Path,
    fill_instructions_path: Optional[Path] = None,
    field_mapping_schema_path: Optional[Path] = None,
    scale: float = 2.0,
) -> None:
    """
    Generate overlays and annotated PNGs: OCR bounding boxes plus boxes/values
    from fill_instructions and field_mapping_schema. Prints each field's bbox + label + value.
    """
    overlays_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    analyses = [AnalysisConfig(name="ocr", path=ocr_analysis_path, color=(255, 0, 255, 160))]
    boxes_by_analysis = build_all_boxes(analyses)

    ocr_fields_by_page: Dict[int, List[dict]] = {}
    if ocr_analysis_path.exists():
        ocr_data = load_json(ocr_analysis_path)
        ocr_fields_by_page = get_ocr_fields_by_page(ocr_data)
    answer_font = _get_bold_dark_blue_font(size=20)
    page_5_font = _get_bold_dark_blue_font(size=10)

    fill_instructions: Optional[List[dict]] = None
    if fill_instructions_path and fill_instructions_path.exists():
        fill_instructions = load_json(fill_instructions_path)
        if not isinstance(fill_instructions, list):
            fill_instructions = None
    fill_by_page = _group_fill_instructions_by_page(fill_instructions or [])

    field_mapping_schema: Optional[dict] = None
    mapping_by_page: Dict[int, List[dict]] = {}
    if field_mapping_schema_path and field_mapping_schema_path.exists():
        field_mapping_schema = load_json(field_mapping_schema_path)
        mapping_by_page = _field_mapping_fields_by_page(field_mapping_schema or {})

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except Exception:
            font = ImageFont.load_default()

    pdf = pdfium.PdfDocument(str(pdf_path))
    num_pages = len(pdf)
    first_page = pdf.get_page(0)
    page_height_pt = float(first_page.get_height())
    box_color = (0, 180, 0, 200)
    text_color = (0, 100, 0, 255)

    for page_index in range(num_pages):
        page_num = page_index + 1
        page = pdf.get_page(page_index)
        width = int(page.get_width() * scale)
        height = int(page.get_height() * scale)

        overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        # OCR boxes and answer text inside each (bold dark blue)
        page_boxes = boxes_by_analysis.get("ocr", {}).get(page_num, [])
        ocr_fields = ocr_fields_by_page.get(page_num, [])
        for idx, (x0, y0, x1, y1) in enumerate(page_boxes):
            x0_s, x1_s = x0 * scale, x1 * scale
            y0_img = (page_height_pt - y0) * scale
            y1_img = (page_height_pt - y1) * scale
            top, bottom = min(y0_img, y1_img), max(y0_img, y1_img)
            # No magenta bounding box; only draw answer text
            if idx < len(ocr_fields) and ocr_fields[idx].get("answer") is not None:
                _draw_answer_in_box(
                    draw, x0_s, top, x1_s, bottom, ocr_fields[idx].get("answer", ""),
                    answer_font, page_num=page_num, page_5_font=page_5_font,
                )

        # Fill instructions / field mapping boxes + text
        page_fill = fill_by_page.get(page_num, [])
        page_mapping = mapping_by_page.get(page_num, [])
        if page_fill or page_mapping:
            _draw_fill_data_on_overlay(
                draw, page_num, page_height_pt, scale,
                page_fill or None, page_mapping if not page_fill else None,
                box_color, text_color, font,
            )

        overlay_path = overlays_dir / f"page_{page_num}_multi_overlay.png"
        overlay.save(overlay_path, "PNG")

        bitmap = page.render(scale=scale)
        base_img = bitmap.to_pil().convert("RGBA")
        combined = Image.alpha_composite(base_img, overlay)
        annotated_path = annotated_dir / f"page_{page_num}_multi_annotated.png"
        combined.save(annotated_path, "PNG")

        print(f"Page {page_num}:")
        print(f"  Overlay   -> {overlay_path}")
        print(f"  Annotated -> {annotated_path}")

        # Print bounding box + values for this page
        for it in page_fill:
            c = it.get("coords", {})
            x, y, w, h = c.get("x"), c.get("y"), c.get("w", 0), c.get("h", 0)
            if c.get("width") is not None:
                w = c["width"]
            if c.get("height") is not None:
                h = c["height"]
            label = it.get("label", "")
            value = it.get("value")
            print(f"    bbox({x},{y},{w},{h})  {it.get('field_id','')}  label={label!r}  value={value!r}")
        if not page_fill and page_mapping:
            for f in page_mapping:
                c = f.get("coords", {})
                x, y = c.get("x"), c.get("y")
                w, h = c.get("w", 0), c.get("h", 0)
                if c.get("width") is not None:
                    w = c["width"]
                if c.get("height") is not None:
                    h = c["height"]
                label = f.get("label_human", f.get("label", ""))
                value = f.get("fill_value")
                print(f"    bbox({x},{y},{w},{h})  {f.get('field_id','')}  label={label!r}  value={value!r}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent

    annotations_dir = project_root / "annotations" / "usda"

    pdf_path = (
        project_root
        / "templates"
        / "usda"
        / "FSA2001_250321V05LC (14).pdf"
    )

    analyses: List[AnalysisConfig] = [
        AnalysisConfig(
            name="ocr",
            path=annotations_dir / "FSA2001_250321V05LC (14)_ocr_analysis.json",
            color=(255, 0, 255, 160),  # magenta
        ),
    ]
    page6_tables_path = annotations_dir / "page6_tables.json"
    if page6_tables_path.exists():
        analyses.append(
            AnalysisConfig(
                name="docling_tables",
                path=page6_tables_path,
                color=(255, 0, 255, 255),  # magenta border for Docling table cell bboxes (page 6)
            )
        )

    overlays_dir = annotations_dir / "overlays"
    annotated_dir = annotations_dir / "annotated"

    generate_multi_analysis_overlays(
        pdf_path=pdf_path,
        analyses=analyses,
        overlays_dir=overlays_dir,
        annotated_dir=annotated_dir,
        scale=2.0,
    )

