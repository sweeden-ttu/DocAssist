#!/usr/bin/env python3
"""
Map form field widgets to nearby OCR text as ``label`` using geometry (no VLM).

Implements the plan: field_coords (PDF lower-left rect) + text boxes in the same
frame, ranked by left-of-field / above-field overlap, distance, and reading order.

Sources for text boxes:
  - ``--pdf`` + Tesseract (default): rasterize the **vector PDF** with pypdfium2,
    then run ``pytesseract.image_to_data``. Use ``--dpi`` or ``--scale`` to increase
    resolution (default ~288 DPI); higher values usually improve OCR on small type.
  - ``--docling-json``: DoclingDocument JSON with TextItem provenance (optional)

Writes updated OCR analysis JSON and optional diagnostics JSON.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

# -----------------------------------------------------------------------------
# PDF geometry (origin bottom-left, y increases upward)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PdfRect:
    """Axis-aligned rectangle in PDF point space."""

    left: float
    bottom: float
    right: float
    top: float

    @property
    def width(self) -> float:
        return max(0.0, self.right - self.left)

    @property
    def height(self) -> float:
        return max(0.0, self.top - self.bottom)

    @property
    def cx(self) -> float:
        return (self.left + self.right) / 2.0

    @property
    def cy(self) -> float:
        return (self.bottom + self.top) / 2.0


def field_coords_to_rect(x: float, y: float, w: float, h: float) -> PdfRect:
    """PDF form widget rect: (x,y) lower-left, width, height."""
    return PdfRect(left=x, bottom=y, right=x + w, top=y + h)


def intersection_area(a: PdfRect, b: PdfRect) -> float:
    il = max(a.left, b.left)
    ir = min(a.right, b.right)
    ib = max(a.bottom, b.bottom)
    it = min(a.top, b.top)
    if il >= ir or ib >= it:
        return 0.0
    return (ir - il) * (it - ib)


def iou(a: PdfRect, b: PdfRect) -> float:
    inter = intersection_area(a, b)
    if inter <= 0:
        return 0.0
    ua = a.width * a.height + b.width * b.height - inter
    return inter / ua if ua > 0 else 0.0


def vertical_overlap_height(a: PdfRect, b: PdfRect) -> float:
    ib = max(a.bottom, b.bottom)
    it = min(a.top, b.top)
    return max(0.0, it - ib)


def horizontal_overlap_width(a: PdfRect, b: PdfRect) -> float:
    il = max(a.left, b.left)
    ir = min(a.right, b.right)
    return max(0.0, ir - il)


# -----------------------------------------------------------------------------
# OCR text boxes → PDF rects (Tesseract image coordinates)
# -----------------------------------------------------------------------------


def tesseract_word_to_pdf_rect(
    ix: int,
    iy: int,
    iw: int,
    ih: int,
    page_width_pt: float,
    page_height_pt: float,
    scale: float,
) -> PdfRect:
    """Convert Tesseract word box (image top-left origin) to PDF rect."""
    l = ix / scale
    r = (ix + iw) / scale
    # Image y down: top of box at iy, bottom at iy+ih
    top_pdf = page_height_pt - iy / scale
    bottom_pdf = page_height_pt - (iy + ih) / scale
    lo, hi = min(bottom_pdf, top_pdf), max(bottom_pdf, top_pdf)
    return PdfRect(left=l, bottom=lo, right=r, top=hi)


def load_tesseract_boxes_for_pages(
    pdf_path: Path,
    page_nums: set[int],
    scale: float,
    tesseract_config: str = r"--oem 3 --psm 6",
) -> dict[int, list[tuple[PdfRect, str]]]:
    import pypdfium2 as pdfium

    try:
        import pytesseract
    except ImportError as e:
        raise SystemExit(
            "pytesseract is required for --pdf mode. pip install pytesseract"
        ) from e

    pdf = pdfium.PdfDocument(str(pdf_path))
    out: dict[int, list[tuple[PdfRect, str]]] = {}

    for p in sorted(page_nums):
        page = pdf.get_page(p - 1)
        pw = page.get_width()
        ph = page.get_height()
        bitmap = page.render(scale=scale)
        img = bitmap.to_pil()
        ocr_result = pytesseract.image_to_data(
            img, config=tesseract_config, output_type=pytesseract.Output.DICT
        )
        words: list[tuple[PdfRect, str]] = []
        n = len(ocr_result["text"])
        for i in range(n):
            text = (ocr_result["text"][i] or "").strip()
            if not text:
                continue
            try:
                conf = int(ocr_result["conf"][i])
            except (ValueError, TypeError):
                conf = 0
            if conf < 0:
                continue
            ix = int(ocr_result["left"][i])
            iy = int(ocr_result["top"][i])
            iw = int(ocr_result["width"][i])
            ih = int(ocr_result["height"][i])
            if iw <= 0 or ih <= 0:
                continue
            r = tesseract_word_to_pdf_rect(ix, iy, iw, ih, pw, ph, scale)
            words.append((r, text))
        out[p] = words

    return out


def load_docling_text_boxes(
    json_path: Path,
) -> dict[int, list[tuple[PdfRect, str]]]:
    """Load TextItem boxes from DoclingDocument JSON; convert to PDF bottom-left."""
    try:
        from docling_core.types.doc import DoclingDocument, TextItem
    except ImportError as e:
        raise SystemExit(
            "docling-core is required for --docling-json. pip install docling-core"
        ) from e

    raw = json_path.read_text(encoding="utf-8")
    doc = DoclingDocument.model_validate_json(raw)
    out: dict[int, list[tuple[PdfRect, str]]] = {}

    for item, _level in doc.iterate_items():
        if not isinstance(item, TextItem):
            continue
        text = (item.text or "").strip()
        if not text:
            continue
        for prov in item.prov:
            pno = prov.page_no
            page = doc.pages.get(pno)
            if page is None:
                continue
            ph = float(page.size.height)
            bbox = prov.bbox
            bl = bbox.to_bottom_left_origin(ph)
            l, b, r, t = bl.as_tuple()
            rect = PdfRect(left=float(l), bottom=float(b), right=float(r), top=float(t))
            out.setdefault(pno, []).append((rect, text))

    return out


# -----------------------------------------------------------------------------
# Label selection heuristics
# -----------------------------------------------------------------------------


def _dist_pdf(a: PdfRect, b: PdfRect) -> float:
    """Distance between rect centers."""
    dx = a.cx - b.cx
    dy = a.cy - b.cy
    return math.hypot(dx, dy)


def _words_in_left_strip(
    field: PdfRect,
    words: list[tuple[PdfRect, str]],
    max_right_of_field: float = 4.0,
    vertical_pad: float = 18.0,
) -> list[tuple[PdfRect, str]]:
    """Words mostly to the left of the field, vertically overlapping the field row."""
    cands: list[tuple[PdfRect, str]] = []
    for rect, w in words:
        if rect.right > field.left + max_right_of_field:
            continue
        if vertical_overlap_height(field, rect) <= 2.0:
            continue
        # Loose vertical band around field vertical span
        if rect.top < field.bottom - vertical_pad or rect.bottom > field.top + vertical_pad:
            continue
        cands.append((rect, w))
    return cands


def _words_in_above_strip(
    field: PdfRect,
    words: list[tuple[PdfRect, str]],
    height_pt: float = 72.0,
    horizontal_pad: float = 120.0,
) -> list[tuple[PdfRect, str]]:
    """Words in a horizontal band above the field top."""
    cands: list[tuple[PdfRect, str]] = []
    band_bottom = field.top
    band_top = field.top + height_pt
    for rect, w in words:
        if rect.bottom < band_bottom - 2 or rect.top > band_top + 2:
            continue
        # Horizontally: overlap or near field
        if rect.right < field.left - horizontal_pad or rect.left > field.right + horizontal_pad:
            continue
        cands.append((rect, w))
    return cands


def _join_reading_order(boxes: list[tuple[PdfRect, str]]) -> str:
    """Sort left-to-right, merge text."""
    if not boxes:
        return ""
    sorted_boxes = sorted(boxes, key=lambda x: (x[0].left, -x[0].top))
    return " ".join(t for _, t in sorted_boxes if t)


def pick_label_for_field(
    field_rect: PdfRect,
    page_words: list[tuple[PdfRect, str]],
) -> tuple[str, float, str, list[dict[str, Any]]]:
    """
    Returns (label, score 0..1, method, debug_candidates).
    """
    debug: list[dict[str, Any]] = []

    left_c = _words_in_left_strip(field_rect, page_words)
    if left_c:
        label = _join_reading_order(left_c)
        # Score: tighter horizontal gap + vertical overlap
        gaps = [field_rect.left - r.right for r, _ in left_c if field_rect.left >= r.right]
        min_gap = min(gaps) if gaps else 0.0
        vov = sum(vertical_overlap_height(field_rect, r) for r, _ in left_c)
        score = min(1.0, 0.35 + 0.4 * min(1.0, vov / max(8.0, field_rect.height)) + 0.25 * math.exp(-min_gap / 40.0))
        for r, t in left_c[:12]:
            debug.append({"text": t, "role": "left", "iou": iou(field_rect, r), "dist": _dist_pdf(field_rect, r)})
        return label.strip(), score, "left_strip", debug

    above_c = _words_in_above_strip(field_rect, page_words)
    if above_c:
        label = _join_reading_order(above_c)
        score = min(1.0, 0.3 + 0.5 * min(1.0, len(above_c) / 5.0))
        for r, t in above_c[:12]:
            debug.append({"text": t, "role": "above", "iou": iou(field_rect, r), "dist": _dist_pdf(field_rect, r)})
        return label.strip(), score, "above_strip", debug

    # Fallback: best IoU among words (weak signal)
    best_iou = 0.0
    best_word = ""
    best_rect: Optional[PdfRect] = None
    for rect, w in page_words:
        v = iou(field_rect, rect)
        if v > best_iou:
            best_iou = v
            best_word = w
            best_rect = rect
    if best_rect is not None and best_iou > 0.01:
        debug.append({"text": best_word, "role": "iou_fallback", "iou": best_iou})
        return best_word, min(0.45, best_iou * 5), "iou_fallback", debug

    # Last resort: nearest word center by distance (within page)
    best_d = float("inf")
    nearest = ""
    for rect, w in page_words:
        d = _dist_pdf(field_rect, rect)
        if d < best_d:
            best_d = d
            nearest = w
    if nearest and best_d < 200:
        debug.append({"text": nearest, "role": "nearest", "dist": best_d})
        return nearest, max(0.05, 1.0 - best_d / 400.0), "nearest", debug

    return "", 0.0, "none", debug


def confidence_from_score(score: float) -> str:
    if score >= 0.45:
        return "high"
    if score >= 0.22:
        return "medium"
    return "low"


def iter_pages_from_ocr_json(data: dict[str, Any]) -> Iterator[tuple[str, int, dict[str, Any]]]:
    for key, page_data in data.items():
        if not key.startswith("page_"):
            continue
        try:
            n = int(key.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if isinstance(page_data, dict):
            yield key, n, page_data


def map_labels(
    ocr_analysis: dict[str, Any],
    boxes_by_page: dict[int, list[tuple[PdfRect, str]]],
    force: bool,
    min_score: float,
) -> tuple[int, list[dict[str, Any]]]:
    """Mutate ocr_analysis fields' label; return (count_updated, diagnostics_rows)."""
    diagnostics: list[dict[str, Any]] = []
    updated = 0

    for _page_key, page_num, page_data in iter_pages_from_ocr_json(ocr_analysis):
        fields = page_data.get("fields") or []
        words = boxes_by_page.get(page_num, [])
        for fi, field in enumerate(fields):
            prev = (field.get("label") or "").strip()
            if prev and not force:
                continue
            coords = field.get("field_coords") or {}
            try:
                x = float(coords["x"])
                y = float(coords["y"])
                w = float(coords["width"])
                h = float(coords["height"])
            except (KeyError, TypeError, ValueError):
                diagnostics.append(
                    {
                        "page": page_num,
                        "field_index": fi,
                        "field": field.get("field"),
                        "error": "bad_field_coords",
                    }
                )
                continue

            frect = field_coords_to_rect(x, y, w, h)
            label, score, method, cand = pick_label_for_field(frect, words)
            conf = confidence_from_score(score)
            diagnostics.append(
                {
                    "page": page_num,
                    "field_index": fi,
                    "field": field.get("field"),
                    "method": method,
                    "score": round(score, 4),
                    "confidence": conf,
                    "label": label,
                    "candidates": cand[:8],
                }
            )
            if label and score >= min_score:
                field["label"] = label
                field["label_confidence"] = conf
                field["label_score"] = round(score, 4)
                field["label_method"] = method
                updated += 1

    return updated, diagnostics


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fill OCR analysis JSON labels from geometry + OCR/Docling text boxes."
    )
    p.add_argument(
        "--ocr-analysis",
        type=Path,
        required=True,
        help="Path to *_ocr_analysis.json (page_N.fields with field_coords).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: overwrite --ocr-analysis).",
    )
    p.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Source PDF (Tesseract mode). Required unless --docling-json is set.",
    )
    p.add_argument(
        "--docling-json",
        type=Path,
        default=None,
        help="DoclingDocument JSON export with TextItem provenance (alternative to --pdf).",
    )
    p.add_argument(
        "--dpi",
        type=float,
        default=None,
        help=(
            "Target raster DPI for PDF render (overrides --scale). "
            "Typical: 300 for print-like OCR. scale = dpi / 72."
        ),
    )
    p.add_argument(
        "--scale",
        type=float,
        default=4.0,
        help=(
            "pypdfium2 render scale (pixels per PDF point). "
            "Approximate DPI ≈ scale × 72 (default 4.0 → ~288 DPI). Ignored if --dpi is set."
        ),
    )
    p.add_argument(
        "--tesseract-config",
        type=str,
        default=r"--oem 3 --psm 6",
        help='Tesseract config string (default: "--oem 3 --psm 6").',
    )
    p.add_argument(
        "--pages",
        type=str,
        default="",
        help="Comma-separated page numbers only (empty = all pages in JSON).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite fields that already have a non-empty label (default: only fill empty).",
    )
    p.add_argument(
        "--min-score",
        type=float,
        default=0.12,
        help="Minimum score to write label (default 0.12).",
    )
    p.add_argument(
        "--diagnostics",
        type=Path,
        default=None,
        help="Write diagnostics JSON (list of per-field mapping details).",
    )
    args = p.parse_args()

    ocr_path = args.ocr_analysis.expanduser().resolve()
    if not ocr_path.is_file():
        raise SystemExit(f"Not found: {ocr_path}")

    with open(ocr_path, encoding="utf-8") as f:
        data = json.load(f)

    page_nums: set[int] = set()
    for _k, n, _pd in iter_pages_from_ocr_json(data):
        page_nums.add(n)
    if args.pages.strip():
        page_nums = {int(x.strip()) for x in args.pages.split(",") if x.strip()}

    ocr_meta: Optional[dict[str, Any]] = None

    if args.docling_json:
        dj_path = args.docling_json.expanduser().resolve()
        boxes = load_docling_text_boxes(dj_path)
        boxes = {p: boxes.get(p, []) for p in page_nums}
        ocr_meta = {"source": "docling-json", "docling_json_path": str(dj_path)}
    elif args.pdf:
        pdf_path = args.pdf.expanduser().resolve()
        if not pdf_path.is_file():
            raise SystemExit(f"PDF not found: {pdf_path}")
        if args.dpi is not None:
            render_scale = float(args.dpi) / 72.0
        else:
            render_scale = float(args.scale)
        approx_dpi = render_scale * 72.0
        print(
            f"Rasterizing PDF for OCR: scale={render_scale:.3f} (~{approx_dpi:.0f} DPI) → {pdf_path.name}"
        )
        boxes = load_tesseract_boxes_for_pages(
            pdf_path,
            page_nums,
            render_scale,
            tesseract_config=args.tesseract_config,
        )
        ocr_meta = {
            "source": "pdf-raster",
            "pdf_path": str(pdf_path),
            "render_scale": round(render_scale, 6),
            "approximate_dpi": round(approx_dpi, 1),
            "tesseract_config": args.tesseract_config,
        }
    else:
        raise SystemExit("Provide --pdf or --docling-json for text boxes.")

    updated, diag = map_labels(
        data,
        boxes,
        force=args.force,
        min_score=args.min_score,
    )

    out_path = args.out or ocr_path
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Updated {updated} field labels → {out_path}")
    if args.diagnostics:
        args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        diag_out: dict[str, Any] = {
            "metadata": ocr_meta or {},
            "fields": diag,
        }
        with open(args.diagnostics, "w", encoding="utf-8") as f:
            json.dump(diag_out, f, indent=2)
        print(f"Diagnostics → {args.diagnostics}")


if __name__ == "__main__":
    main()
