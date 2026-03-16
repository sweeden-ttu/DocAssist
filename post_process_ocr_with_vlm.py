import argparse
import base64
import io
import json
import logging
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from PIL import Image, ImageFilter
from PIL.ImageOps import crop

# Defer docling imports when running --fix-labels, --apply-data, or --annotate-with-vlm to avoid loading MLX/Metal (can crash in some envs)
_USE_LIGHTWEIGHT_MODE = (
    "--fix-labels" in sys.argv
    or "--apply-data" in sys.argv
    or "--annotate-with-vlm" in sys.argv
)
if not _USE_LIGHTWEIGHT_MODE:
    from docling_core.types.doc import (
        DoclingDocument,
        ImageRefMode,
        NodeItem,
        TextItem,
    )
    from docling_core.types.doc.document import (
        ContentLayer,
        DocItem,
        FormItem,
        GraphCell,
        KeyValueItem,
        PictureItem,
        RichTableCell,
        TableCell,
        TableItem,
    )
    from pydantic import BaseModel, ConfigDict
    from tqdm import tqdm

    from docling.backend.json.docling_json_backend import DoclingJSONBackend
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat, ItemAndImageEnrichmentElement
    from docling.datamodel.document import ConversionResult
    from docling.datamodel.pipeline_options import (
        ConvertPipelineOptions,
        PdfPipelineOptions,
        PictureDescriptionApiOptions,
    )
    from docling.document_converter import DocumentConverter, FormatOption, PdfFormatOption
    from docling.exceptions import OperationNotAllowed
    from docling.models.base_model import BaseModelWithOptions, GenericEnrichmentModel
    from docling.pipeline.simple_pipeline import SimplePipeline
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
    from docling.utils.api_image_request import api_image_request
    from docling.utils.profiling import ProfilingScope, TimeRecorder
    from docling.utils.utils import chunkify

# Example on how to apply to Docling Document OCR as a post-processing with "nanonets-ocr2-3b" via LM Studio
# Requires LM Studio running inference server with "nanonets-ocr2-3b" model pre-loaded
# To run:
# uv run python docs/examples/post_process_ocr_with_vlm.py

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_STUDIO_MODEL = "nanonets-ocr2-3b"

DEFAULT_PROMPT = "Extract the text from the above document as if you were reading it naturally. Output pure text, no html and no markdown. Pay attention on line breaks and don't miss text after line break. Put all text in one line."
VERBOSE = True
SHOW_IMAGE = False
SHOW_EMPTY_CROPS = False
SHOW_NONEMPTY_CROPS = False
PRINT_RESULT_MARKDOWN = False


def is_empty_fast_with_lines_pil(
    pil_img: Image.Image,
    downscale_max_side: int = 48,  # 64
    grad_threshold: float = 15.0,  # how strong a gradient must be to count as edge
    min_line_coverage: float = 0.6,  # line must cover 60% of height/width
    max_allowed_lines: int = 10,  # allow up to this many strong lines (default 4)
    edge_fraction_threshold: float = 0.0035,
):
    """
    Fast 'empty' detector using only PIL + NumPy.

    Treats an image as empty if:
      - It has very few edges overall, OR
      - Edges can be explained by at most `max_allowed_lines` long vertical/horizontal lines.

    Returns:
      (is_empty: bool, remaining_edge_fraction: float, debug: dict)
    """

    # 1) Convert to grayscale
    gray = pil_img.convert("L")

    # 2) Aggressive downscale, keeping aspect ratio
    w0, h0 = gray.size
    max_side = max(w0, h0)
    if max_side > downscale_max_side:
        # scale = downscale_max_side / max_side
        # new_w = max(1, int(w0 * scale))
        # new_h = max(1, int(h0 * scale))

        new_w = downscale_max_side
        new_h = downscale_max_side

        gray = gray.resize((new_w, new_h), resample=Image.BILINEAR)

    w, h = gray.size
    if w == 0 or h == 0:
        return True, 0.0, {"reason": "zero_size"}

    # 3) Small blur to reduce noise
    gray = gray.filter(ImageFilter.BoxBlur(1))

    # 4) Convert to NumPy
    arr = np.asarray(
        gray, dtype=np.float32
    )  # shape (h, w) in PIL, but note: PIL size is (w, h)
    H, W = arr.shape

    # 5) Compute simple gradients (forward differences)
    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)

    gx[:, :-1] = arr[:, 1:] - arr[:, :-1]  # horizontal differences
    gy[:-1, :] = arr[1:, :] - arr[:-1, :]  # vertical differences

    mag = np.hypot(gx, gy)  # gradient magnitude

    # 6) Threshold gradients to get edges (boolean mask)
    edges = mag > grad_threshold
    edge_fraction = edges.mean()

    # Quick early-exit: almost no edges => empty
    if edge_fraction < edge_fraction_threshold:
        return True, float(edge_fraction), {"reason": "few_edges"}

    # 7) Detect strong vertical & horizontal lines via edge sums
    col_sum = edges.sum(axis=0)  # per column
    row_sum = edges.sum(axis=1)  # per row

    # Line must have edge pixels in at least `min_line_coverage` of the dimension
    vert_line_cols = np.where(col_sum >= min_line_coverage * H)[0]
    horiz_line_rows = np.where(row_sum >= min_line_coverage * W)[0]

    num_lines = len(vert_line_cols) + len(horiz_line_rows)

    # If we have more long lines than allowed => non-empty
    if num_lines > max_allowed_lines:
        return (
            False,
            float(edge_fraction),
            {
                "reason": "too_many_lines",
                "num_lines": int(num_lines),
                "edge_fraction": float(edge_fraction),
            },
        )

    # 8) Mask out those lines and recompute remaining edges
    line_mask = np.zeros_like(edges, dtype=bool)
    if len(vert_line_cols) > 0:
        line_mask[:, vert_line_cols] = True
    if len(horiz_line_rows) > 0:
        line_mask[horiz_line_rows, :] = True

    remaining_edges = edges & ~line_mask
    remaining_edge_fraction = remaining_edges.mean()

    is_empty = remaining_edge_fraction < edge_fraction_threshold

    debug = {
        "original_edge_fraction": float(edge_fraction),
        "remaining_edge_fraction": float(remaining_edge_fraction),
        "num_vert_lines": len(vert_line_cols),
        "num_horiz_lines": len(horiz_line_rows),
    }
    return is_empty, float(remaining_edge_fraction), debug


def remove_break_lines(text: str) -> str:
    # Replace any newline types with a single space
    cleaned = re.sub(r"[\r\n]+", " ", text)
    # Collapse multiple spaces into one
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def safe_crop(img: Image.Image, bbox):
    left, top, right, bottom = bbox
    # Clamp to image boundaries
    left = max(0, min(left, img.width))
    top = max(0, min(top, img.height))
    right = max(0, min(right, img.width))
    bottom = max(0, min(bottom, img.height))
    return img.crop((left, top, right, bottom))


def no_long_repeats(s: str, threshold: int) -> bool:
    """
    Returns False if the string `s` contains more than `threshold`
    identical characters in a row, otherwise True.
    """
    pattern = r"(.)\1{" + str(threshold) + ",}"
    return re.search(pattern, s) is None


# Prompt for extracting the form label from a crop of the region above a field bbox
LABEL_EXTRACTION_PROMPT = (
    "This image shows the label or title of a form field (the text directly above an input box) "
    "on the FSA-2001 Request for Direct Loan Assistance form. "
    "Output the most descriptive field label possible in this exact format: Part <letter>, # <description of field label>. "
    "Example: Part A, 1. Exact Full Legal Name. "
    "As an individual applicant, the parts to complete are Part(s) B, E, F, G, H, I, J, L; "
    "include the Part letter and item number when the form shows them. "
    "Output a single line only, no explanation, no markdown. "
    "Give the full, descriptive label text (e.g. 'Part B, 1. Social Security Number (9 Digits)' not just 'SSN')."
)


def _lm_studio_image_request(
    image: Image.Image,
    prompt: str,
    url: str,
    model: str,
    timeout: int = 30,
    max_tokens: int = 256,
) -> str:
    """Call LM Studio chat completions API with one image (no docling dependency)."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        body = ""
        if e.fp:
            try:
                body = e.fp.read().decode("utf-8", errors="replace")
            except Exception:
                pass
        msg = f"LM Studio HTTP {e.code}: {e.reason}"
        if body.strip():
            msg += f" — {body.strip()[:500]}"
        raise RuntimeError(msg) from e
    choices = data.get("choices", [])
    if not choices:
        return ""
    return (choices[0].get("message") or {}).get("content", "").strip()


def _crop_label_region_above_field(
    page_img: Image.Image,
    page_width_pt: float,
    page_height_pt: float,
    field_x_pt: float,
    field_y_pt: float,
    field_width_pt: float,
    field_height_pt: float,
    scale: float,
    label_height_pt: float = 55.0,
    expand_horizontal_pt: float = 30.0,
) -> Optional[Image.Image]:
    """
    Crop the page image to the region directly above the field bbox (the form label).
    PDF coords: origin bottom-left, y increases upward. Field rect: (x, y, width, height).
    """
    # Label strip in PDF: above the field top, same horizontal extent (slightly expanded)
    left_pt = max(0, field_x_pt - expand_horizontal_pt)
    right_pt = min(
        page_width_pt,
        field_x_pt + field_width_pt + expand_horizontal_pt,
    )
    # Field top in PDF = field_y_pt + field_height_pt. Label strip: from field top to field top + label_height_pt
    bottom_pt = field_y_pt + field_height_pt
    top_pt = bottom_pt + label_height_pt

    # Convert to image coords (origin top-left, y down). Scale by scale
    img_h = page_img.height
    img_w = page_img.width
    # PDF (left, bottom, right, top) -> image (left, top, right, bottom)
    im_left = int(left_pt * scale)
    im_right = int(right_pt * scale)
    # image y: top of label strip = page_height_pt - top_pt (PDF), then * scale
    im_top = int((page_height_pt - top_pt) * scale)
    im_bottom = int((page_height_pt - bottom_pt) * scale)

    im_left = max(0, min(im_left, img_w))
    im_right = max(0, min(im_right, img_w))
    im_top = max(0, min(im_top, img_h))
    im_bottom = max(0, min(im_bottom, img_h))

    if im_left >= im_right or im_top >= im_bottom:
        return None
    return page_img.crop((im_left, im_top, im_right, im_bottom))


def _get_value_at_path(data: dict, path: str) -> Any:
    """Get a value from nested dict using dot-separated path (e.g. 'part_a_applicant.full_legal_name')."""
    keys = path.strip().split(".")
    obj: Any = data
    for k in keys:
        if isinstance(obj, dict) and k in obj:
            obj = obj[k]
        else:
            return None
    return obj


def _format_value_for_field(value: Any) -> str:
    """Format a data value for display in OCR answer (e.g. date, number). Skip list/dict."""
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return ""
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value == int(value):
            return str(int(value))
        return str(value)
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


# Default mapping: page -> { OCR field_id -> data path } for applying data_clean_2 to FSA2001 pages 6-9.
# Extend or override with --mapping JSON file: { "6": { "TextField18[0]": "part_a_applicant.full_legal_name" }, ... }
DEFAULT_APPLY_MAPPING: dict[int, dict[str, str]] = {
    6: {
        "TextField18[0]": "part_a_applicant.full_legal_name",
        "DateTimeField1[0]": "part_h_balance_sheet.as_of_date",
        "H_TotalIntermFarmAssets": "part_h_balance_sheet.farm_assets.intermediate.total_intermediate",
        "H_TotalLTFarmAssets": "part_h_balance_sheet.farm_assets.long_term.total_long_term",
        "H_TotalCurrentPersonalAssets": "part_h_balance_sheet.personal_assets.current.total_current",
        "H_TotalIntermPersonalAssets": "part_h_balance_sheet.personal_assets.intermediate.total_intermediate",
    },
    7: {
        "5_MarketValue[0]": "part_h_balance_sheet.farm_assets.intermediate.total_intermediate",
        "4_MarketValue[0]": "part_h_balance_sheet.farm_assets.long_term.real_estate_land",
        "SchedN_Savings": "part_h_balance_sheet.personal_assets.current.savings",
        "SchedN_Total": "part_h_balance_sheet.personal_assets.current.total_current",
        "SchedM_FarmName": "farm_address.description",
        "SchedM_TotalAcres": "farm_address.total_acres",
        "SchedM_MarketValue": "part_h_balance_sheet.farm_assets.long_term.real_estate_land",
    },
    8: {},
    9: {},
}


def apply_data_to_ocr_analysis(
    data: dict,
    ocr_analysis: dict,
    pages: list[int],
    mapping: dict[int, dict[str, str]],
) -> int:
    """
    Merge values from `data` into OCR analysis `answer` fields for the given pages using
    (page -> field_id -> data_path) mapping. Modifies ocr_analysis in place. Returns count updated.
    """
    updated = 0
    for page_num in pages:
        page_key = f"page_{page_num}"
        page_data = ocr_analysis.get(page_key)
        if not page_data or not isinstance(page_data, dict):
            continue
        fields = page_data.get("fields", [])
        page_map = mapping.get(page_num, {})
        for field in fields:
            field_id = field.get("field")
            if not field_id or field_id not in page_map:
                continue
            path = page_map[field_id]
            value = _get_value_at_path(data, path)
            if value is None:
                continue
            answer = _format_value_for_field(value)
            if answer:
                field["answer"] = answer
                updated += 1
    return updated


def _flatten_data_for_vlm(obj: Any, prefix: str = "") -> list[str]:
    """Flatten nested dict/list into 'path: value' lines (scalars only) for VLM context."""
    lines: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                lines.extend(_flatten_data_for_vlm(v, key))
            else:
                lines.append(f"{key}: {_format_value_for_field(v)}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]" if prefix else f"[{i}]"
            if isinstance(v, (dict, list)):
                lines.extend(_flatten_data_for_vlm(v, key))
            else:
                lines.append(f"{key}: {_format_value_for_field(v)}")
    return lines


# Parts B, E, F, G, H, I, J, L should be filled; on page 6 only Part H subsections 2A–2F
PARTS_TO_FILL = ("B", "E", "F", "G", "H", "I", "J", "L")
PART_H_PAGE6_SUBSECTIONS = ("2A", "2B", "2C", "2D", "2E", "2F")


def _field_is_in_scope(
    page_num: int,
    label: str,
    parts_to_fill: tuple[str, ...] = PARTS_TO_FILL,
    part_h_page6_subsections: tuple[str, ...] = PART_H_PAGE6_SUBSECTIONS,
) -> bool:
    """True if this field should be filled: part in B,E,F,G,H,I,J,L; on page 6 only Part H 2A–2F; pages 7–8 all fields."""
    if not label or not isinstance(label, str):
        return False
    # Pages 7 and 8: all fields in scope (fill with 0 or N/A)
    if page_num in (7, 8):
        return True
    label_upper = label.upper()
    # Must mention one of the fill parts (e.g. "Part B", "Part H,")
    part_match = False
    for p in parts_to_fill:
        if f"PART {p}" in label_upper or f"PART {p}," in label_upper or f"PART {p} " in label_upper:
            part_match = True
            break
    if not part_match:
        return False
    # On page 6, Part H only: restrict to subsections 2A–2F
    if page_num == 6 and "PART H" in label_upper:
        for sub in part_h_page6_subsections:
            if f", {sub}" in label_upper or f", {sub}." in label_upper or f" {sub} " in label_upper:
                return True
        return False
    return True


# Keep short to stay under 4K context with image + data
ANNOTATE_FIELDS_VLM_PROMPT = """Form page image + two lists. LIST 1 = fields (index. label). LIST 2 = data (path: value).
Fill empty fields from LIST 2. Output one line per field: INDEX: value (use 0-based index). Exact value from data when possible. Optional/blank: INDEX: (leave blank). No other text."""


def _parse_vlm_annotation_response(text: str, num_fields: int) -> dict[int, str]:
    """Parse VLM response into field_index -> value. Expects lines like '0: value' or '0. value'."""
    result: dict[int, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Match "0: value" or "0. value" or "0) value"
        m = re.match(r"^\s*(\d+)\s*[.:)]\s*(.*)$", line)
        if m:
            idx = int(m.group(1))
            val = m.group(2).strip()
            if idx < num_fields and val.lower() not in ("(leave blank)", "leave blank", "n/a", ""):
                result[idx] = val
            elif idx < num_fields and val:
                # Allow explicit blank
                result[idx] = ""
    return result


# Max data lines and label/value lengths to stay under ~4K context with image
_ANNOTATE_MAX_DATA_LINES = 50
_ANNOTATE_MAX_LABEL_LEN = 72
_ANNOTATE_MAX_VALUE_LEN = 48


def _annotate_one_page_with_vlm(
    page_img: Image.Image,
    fields: list[dict],
    data_lines: list[str],
    url: str = LM_STUDIO_URL,
    model: str = LM_STUDIO_MODEL,
    timeout: int = 60,
) -> dict[int, str]:
    """
    Send one page image + field list + data to LM Studio; return dict of field_index -> value.
    Keeps prompt + data small to fit 4K context.
    """
    field_blobs = []
    for i, f in enumerate(fields):
        label = (f.get("label") or "").strip()
        if len(label) > _ANNOTATE_MAX_LABEL_LEN:
            label = label[: _ANNOTATE_MAX_LABEL_LEN - 2] + ".."
        ans = (f.get("answer") or "").strip()
        if len(ans) > _ANNOTATE_MAX_VALUE_LEN:
            ans = ans[: _ANNOTATE_MAX_VALUE_LEN - 2] + ".."
        if ans:
            field_blobs.append(f"{i}. {label} ({ans})")
        else:
            field_blobs.append(f"{i}. {label} [empty]")
    # Cap data lines and truncate long values to stay under context
    truncated = []
    for line in data_lines[:_ANNOTATE_MAX_DATA_LINES]:
        if len(line) > 100:
            line = line[:97] + ".."
        truncated.append(line)
    data_blob = "\n".join(truncated)
    if len(data_lines) > _ANNOTATE_MAX_DATA_LINES:
        data_blob += "\n..."
    prompt = (
        ANNOTATE_FIELDS_VLM_PROMPT
        + "\n\nLIST 1:\n"
        + "\n".join(field_blobs)
        + "\n\nLIST 2:\n"
        + data_blob
        + "\n\nOutput INDEX: value per field:"
    )
    response = _lm_studio_image_request(
        image=page_img,
        prompt=prompt,
        url=url,
        model=model,
        timeout=timeout,
        max_tokens=1024,
    )
    return _parse_vlm_annotation_response(response, len(fields))


def run_annotate_pages_with_vlm_iterative(
    data_path: Path,
    ocr_analysis_path: Path,
    pdf_path: Path,
    pages: list[int],
    max_iterations: int = 3,
    url: str = LM_STUDIO_URL,
    model: str = LM_STUDIO_MODEL,
    scale: float = 1.0,
    save_after: bool = True,
) -> None:
    """
    Iteratively call LM Studio with the form page images and applicant data until all fields
    on the given pages are annotated (or max_iterations reached). Updates ocr_analysis in place
    and saves when done. Requires pypdfium2 for PDF rendering.
    Uses scale=1.0 by default to keep image small enough for 4K context.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise SystemExit("annotate-with-vlm requires pypdfium2. Install with: pip install pypdfium2")

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    with open(ocr_analysis_path, encoding="utf-8") as f:
        ocr_analysis = json.load(f)

    data_lines = _flatten_data_for_vlm(data)
    pdf = pdfium.PdfDocument(str(pdf_path))
    page_images: dict[int, Image.Image] = {}
    page_height_pt = 792.0

    def get_page_image(page_num: int) -> Image.Image:
        nonlocal page_height_pt
        if page_num not in page_images:
            page = pdf.get_page(page_num - 1)
            page_height_pt = float(page.get_height())
            w = int(page.get_width() * scale)
            h = int(page.get_height() * scale)
            bitmap = page.render(scale=scale)
            page_images[page_num] = bitmap.to_pil()
        return page_images[page_num]

    for page_num in pages:
        page_key = f"page_{page_num}"
        page_data = ocr_analysis.get(page_key)
        if not page_data or not isinstance(page_data, dict):
            print(f"  Skip {page_key}: no page data")
            continue
        fields = page_data.get("fields", [])
        if not fields:
            print(f"  Skip {page_key}: no fields")
            continue

        # Only fill Parts B,E,F,G,H,I,J,L; on page 6 only Part H 2A,2B,2C,2E,2F
        in_scope_indices = [
            i for i, f in enumerate(fields)
            if _field_is_in_scope(page_num, f.get("label") or "")
        ]
        if not in_scope_indices:
            print(f"  Skip {page_key}: no in-scope fields (Parts B,E,F,G,H,I,J,L; page 6 only Part H 2A–2F).")
            continue

        img = get_page_image(page_num)
        iteration = 0
        while iteration < max_iterations:
            # Only consider in-scope fields as missing; only send page if some are missing
            missing_in_scope = [
                i for i in in_scope_indices
                if not (fields[i].get("answer") or "").strip()
            ]
            if not missing_in_scope:
                print(f"  Page {page_num}: all in-scope fields annotated.")
                break
            print(f"  Page {page_num} iteration {iteration + 1}: {len(missing_in_scope)} in-scope field(s) missing.")
            # Send only in-scope fields to VLM (indices 0..n-1)
            fields_to_send = [fields[i] for i in in_scope_indices]
            try:
                updates = _annotate_one_page_with_vlm(
                    page_img=img,
                    fields=fields_to_send,
                    data_lines=data_lines,
                    url=url,
                    model=model,
                )
            except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError, OSError) as e:
                print(f"  VLM request failed: {e}")
                iteration += 1
                continue
            if not updates:
                iteration += 1
                continue
            # Map local index back to actual field index
            for local_idx, value in updates.items():
                if local_idx < len(in_scope_indices):
                    actual_idx = in_scope_indices[local_idx]
                    fields[actual_idx]["answer"] = value
                    if value:
                        print(f"    Field {actual_idx} ({fields[actual_idx].get('field', '')}): {value[:50]!r}...")
            iteration += 1

        if iteration >= max_iterations:
            still = [
                i for i in in_scope_indices
                if not (fields[i].get("answer") or "").strip()
            ]
            if still:
                print(f"  Page {page_num}: {len(still)} in-scope field(s) still empty after {max_iterations} iterations.")

        # Pages 7 and 8: set any remaining empty in-scope field to 0 or N/A
        if page_num in (7, 8):
            for i in in_scope_indices:
                ans = (fields[i].get("answer") or "").strip()
                if not ans:
                    label = (fields[i].get("label") or "").upper()
                    fields[i]["answer"] = "0" if ("$" in label or "VALUE" in label or "AMOUNT" in label or "ACRES" in label) else "N/A"
                    print(f"  Page {page_num} field {i}: set empty to {fields[i]['answer']!r}")

    if save_after:
        with open(ocr_analysis_path, "w", encoding="utf-8") as f:
            json.dump(ocr_analysis, f, indent=2)
        print(f"Saved {ocr_analysis_path}")


def run_apply_data_and_overlays(
    data_path: Path,
    ocr_analysis_path: Path,
    pdf_path: Path,
    pages: list[int],
    mapping_path: Optional[Path] = None,
    overlays_dir: Optional[Path] = None,
    annotated_dir: Optional[Path] = None,
    prompt_before_apply: bool = True,
) -> None:
    """
    Load data JSON and OCR analysis, optionally prompt user, apply data to OCR for given pages,
    save updated OCR analysis, then run generate_multi_analysis_overlays to produce annotated PNGs.
    """
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    with open(ocr_analysis_path, encoding="utf-8") as f:
        ocr_analysis = json.load(f)

    mapping: dict[int, dict[str, str]] = {}
    if mapping_path and mapping_path.exists():
        with open(mapping_path, encoding="utf-8") as f:
            raw = json.load(f)
        for k, v in raw.items():
            try:
                pn = int(k)
                if isinstance(v, dict):
                    mapping[pn] = v
            except (ValueError, TypeError):
                continue
    if not mapping:
        mapping = dict(DEFAULT_APPLY_MAPPING)

    if prompt_before_apply:
        print(
            f"Apply data from {data_path.name} to OCR analysis {ocr_analysis_path.name} for pages {sorted(pages)}?"
        )
        print(f"  Mapping: {sum(len(m) for m in mapping.values())} field(s) defined.")
        reply = input("Proceed? (y/n): ").strip().lower()
        if reply != "y" and reply != "yes":
            print("Aborted.")
            return

    n = apply_data_to_ocr_analysis(data, ocr_analysis, pages, mapping)
    print(f"Updated {n} field(s) in OCR analysis.")
    with open(ocr_analysis_path, "w", encoding="utf-8") as f:
        json.dump(ocr_analysis, f, indent=2)
    print(f"Saved {ocr_analysis_path}")

    annotations_dir = ocr_analysis_path.resolve().parent
    overlays_dir = overlays_dir or (annotations_dir / "overlays")
    annotated_dir = annotated_dir or (annotations_dir / "annotated")
    overlays_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    # Run overlay generation (same entry point as pipeline: OCR boxes + answers)
    try:
        from generate_multi_analysis_overlays import run_overlays_with_fill_data
    except ImportError:
        raise SystemExit(
            "apply-data requires generate_multi_analysis_overlays. Run from DocAssist project root."
        )
    run_overlays_with_fill_data(
        pdf_path=pdf_path,
        ocr_analysis_path=ocr_analysis_path,
        overlays_dir=overlays_dir,
        annotated_dir=annotated_dir,
    )
    print("Annotated pages written to", annotated_dir)


def fix_ocr_analysis_labels(
    pdf_path: Path,
    ocr_analysis_path: Path,
    out_path: Optional[Path] = None,
    url: str = LM_STUDIO_URL,
    prompt: str = LABEL_EXTRACTION_PROMPT,
    scale: float = 2.0,
    label_height_pt: float = 55.0,
    concurrency: int = 2,
) -> Path:
    """
    Load OCR analysis JSON (from ocr_field_analysis.py), and for each field replace
    the "label" with the most descriptive label that the bounding box is directly
    below: crop the region above each field, run VLM OCR on that crop, and set
    field["label"] to the result. Saves the updated JSON to out_path (default:
    overwrite ocr_analysis_path).
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise SystemExit(
            "fix-labels mode requires pypdfium2. Install with: pip install pypdfium2"
        )

    out_path = out_path or ocr_analysis_path
    with open(ocr_analysis_path, encoding="utf-8") as f:
        data = json.load(f)

    pdf = pdfium.PdfDocument(str(pdf_path))
    page_images: dict[int, Image.Image] = {}

    def get_page_image(page_num: int) -> Image.Image:
        if page_num not in page_images:
            page = pdf.get_page(page_num - 1)
            w = int(page.get_width() * scale)
            h = int(page.get_height() * scale)
            bitmap = page.render(scale=scale)
            page_images[page_num] = bitmap.to_pil()
        return page_images[page_num]

    total_updated = 0
    for page_key, page_data in data.items():
        if not page_key.startswith("page_"):
            continue
        try:
            page_num = int(page_key.split("_")[1])
        except (IndexError, ValueError):
            continue

        page_size = page_data.get("page_size", {})
        page_height_pt = float(page_size.get("height", 792))
        page_width_pt = float(page_size.get("width", 612))

        img = get_page_image(page_num)
        fields = page_data.get("fields", [])
        if not fields:
            continue

        # Collect (field index, crop) for fields that have coords
        crops_and_indices: list[tuple[int, Image.Image]] = []
        for i, field in enumerate(fields):
            coords = field.get("field_coords") or {}
            x = coords.get("x")
            y = coords.get("y")
            w = coords.get("width")
            h = coords.get("height")
            if None in (x, y, w, h):
                continue
            crop = _crop_label_region_above_field(
                img,
                page_width_pt,
                page_height_pt,
                float(x),
                float(y),
                float(w),
                float(h),
                scale=scale,
                label_height_pt=label_height_pt,
            )
            if crop is not None and crop.width > 2 and crop.height > 2:
                crops_and_indices.append((i, crop))

        if not crops_and_indices:
            continue

        # Batch VLM requests for this page
        indices = [idx for idx, _ in crops_and_indices]
        crops = [c for _, c in crops_and_indices]

        def _request_one(crop_img: Image.Image) -> str:
            return _lm_studio_image_request(
                image=crop_img,
                prompt=prompt,
                url=url,
                model=LM_STUDIO_MODEL,
                timeout=30,
            )

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            new_labels = list(executor.map(_request_one, crops))

        for idx, new_label in zip(indices, new_labels):
            cleaned = remove_break_lines(new_label).strip()
            if cleaned and no_long_repeats(cleaned, 50):
                data[page_key]["fields"][idx]["label"] = cleaned
                total_updated += 1
                if VERBOSE:
                    print(f"  {page_key} field {data[page_key]['fields'][idx].get('field', '?')}: {cleaned!r}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Updated {total_updated} labels; saved to {out_path}")
    return out_path


if not _USE_LIGHTWEIGHT_MODE:
    class PostOcrEnrichmentElement(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        item: Union[DocItem, TableCell, RichTableCell, GraphCell]
        image: list[
            Image.Image
        ]  # Needs to be an a list of images for multi-provenance elements


    class PostOcrEnrichmentPipelineOptions(ConvertPipelineOptions):
        api_options: PictureDescriptionApiOptions


    class PostOcrEnrichmentPipeline(SimplePipeline):
        def __init__(self, pipeline_options: PostOcrEnrichmentPipelineOptions):
            super().__init__(pipeline_options)
            self.pipeline_options: PostOcrEnrichmentPipelineOptions

            self.enrichment_pipe = [
                PostOcrApiEnrichmentModel(
                    enabled=True,
                    enable_remote_services=True,
                    artifacts_path=None,
                    options=self.pipeline_options.api_options,
                    accelerator_options=AcceleratorOptions(),
                )
            ]

        @classmethod
        def get_default_options(cls) -> PostOcrEnrichmentPipelineOptions:
            return PostOcrEnrichmentPipelineOptions()

        def _enrich_document(self, conv_res: ConversionResult) -> ConversionResult:
            def _prepare_elements(
                conv_res: ConversionResult, model: GenericEnrichmentModel[Any]
            ) -> Iterable[NodeItem]:
                for doc_element, _level in conv_res.document.iterate_items(
                    traverse_pictures=True,
                    included_content_layers={
                        ContentLayer.BODY,
                        ContentLayer.FURNITURE,
                    },
                ):  # With all content layers, with traverse_pictures=True
                    prepared_elements = (
                        model.prepare_element(  # make this one yield multiple items.
                            conv_res=conv_res, element=doc_element
                        )
                    )
                    if prepared_elements is not None:
                        yield prepared_elements

            with TimeRecorder(conv_res, "doc_enrich", scope=ProfilingScope.DOCUMENT):
                for model in self.enrichment_pipe:
                    for element_batch in chunkify(
                        _prepare_elements(conv_res, model),
                        model.elements_batch_size,
                    ):
                        for element in model(
                            doc=conv_res.document, element_batch=element_batch
                        ):  # Must exhaust!
                            pass
            return conv_res


    class PostOcrApiEnrichmentModel(
        GenericEnrichmentModel[PostOcrEnrichmentElement], BaseModelWithOptions
    ):
        expansion_factor: float = 0.001

        def prepare_element(
            self, conv_res: ConversionResult, element: NodeItem
        ) -> Optional[list[PostOcrEnrichmentElement]]:
            if not self.is_processable(doc=conv_res.document, element=element):
                return None

            allowed = (DocItem, TableItem, GraphCell)
            assert isinstance(element, allowed)

            if isinstance(element, KeyValueItem | FormItem):
                # Yield from the graphCells inside here.
                result = []
                for c in element.graph.cells:
                    element_prov = c.prov  # Key / Value have only one provenance!
                    bbox = element_prov.bbox
                    page_ix = element_prov.page_no
                    bbox = bbox.scale_to_size(
                        old_size=conv_res.document.pages[page_ix].size,
                        new_size=conv_res.document.pages[page_ix].image.size,
                    )
                    expanded_bbox = bbox.expand_by_scale(
                        x_scale=self.expansion_factor, y_scale=self.expansion_factor
                    ).to_top_left_origin(
                        page_height=conv_res.document.pages[page_ix].image.size.height
                    )

                    good_bbox = True
                    if (
                        expanded_bbox.l > expanded_bbox.r
                        or expanded_bbox.t > expanded_bbox.b
                    ):
                        good_bbox = False

                    if good_bbox:
                        cropped_image = conv_res.document.pages[
                            page_ix
                        ].image.pil_image.crop(expanded_bbox.as_tuple())

                        is_empty, rem_frac, debug = is_empty_fast_with_lines_pil(
                            cropped_image
                        )
                        if is_empty:
                            if SHOW_EMPTY_CROPS:
                                try:
                                    cropped_image.show()
                                except Exception as e:
                                    print(f"Error with image: {e}")
                            print(
                                f"Detected empty form item image crop: {rem_frac} - {debug}"
                            )
                        else:
                            result.append(
                                PostOcrEnrichmentElement(item=c, image=[cropped_image])
                            )
                return result
            elif isinstance(element, TableItem):
                element_prov = element.prov[0]
                page_ix = element_prov.page_no
                result = []
                for i, row in enumerate(element.data.grid):
                    for j, cell in enumerate(row):
                        if hasattr(cell, "bbox"):
                            if cell.bbox:
                                bbox = cell.bbox
                                bbox = bbox.scale_to_size(
                                    old_size=conv_res.document.pages[page_ix].size,
                                    new_size=conv_res.document.pages[page_ix].image.size,
                                )

                                expanded_bbox = bbox.expand_by_scale(
                                    x_scale=self.table_cell_expansion_factor,
                                    y_scale=self.table_cell_expansion_factor,
                                ).to_top_left_origin(
                                    page_height=conv_res.document.pages[
                                        page_ix
                                    ].image.size.height
                                )

                                good_bbox = True
                                if (
                                    expanded_bbox.l > expanded_bbox.r
                                    or expanded_bbox.t > expanded_bbox.b
                                ):
                                    good_bbox = False

                                if good_bbox:
                                    cropped_image = conv_res.document.pages[
                                        page_ix
                                    ].image.pil_image.crop(expanded_bbox.as_tuple())

                                    is_empty, rem_frac, debug = (
                                        is_empty_fast_with_lines_pil(cropped_image)
                                    )
                                    if is_empty:
                                        if SHOW_EMPTY_CROPS:
                                            try:
                                                cropped_image.show()
                                            except Exception as e:
                                                print(f"Error with image: {e}")
                                        print(
                                            f"Detected empty table cell image crop: {rem_frac} - {debug}"
                                        )
                                    else:
                                        if SHOW_NONEMPTY_CROPS:
                                            cropped_image.show()
                                        result.append(
                                            PostOcrEnrichmentElement(
                                                item=cell, image=[cropped_image]
                                            )
                                        )
                return result
            else:
                multiple_crops = []
                # Crop the image form the page
                for element_prov in element.prov:
                    # Iterate over provenances
                    bbox = element_prov.bbox

                    page_ix = element_prov.page_no
                    bbox = bbox.scale_to_size(
                        old_size=conv_res.document.pages[page_ix].size,
                        new_size=conv_res.document.pages[page_ix].image.size,
                    )
                    expanded_bbox = bbox.expand_by_scale(
                        x_scale=self.expansion_factor, y_scale=self.expansion_factor
                    ).to_top_left_origin(
                        page_height=conv_res.document.pages[page_ix].image.size.height
                    )

                    good_bbox = True
                    if (
                        expanded_bbox.l > expanded_bbox.r
                        or expanded_bbox.t > expanded_bbox.b
                    ):
                        good_bbox = False

                    if hasattr(element, "text"):
                        if good_bbox:
                            cropped_image = conv_res.document.pages[
                                page_ix
                            ].image.pil_image.crop(expanded_bbox.as_tuple())

                            is_empty, rem_frac, debug = is_empty_fast_with_lines_pil(
                                cropped_image
                            )
                            if is_empty:
                                if SHOW_EMPTY_CROPS:
                                    try:
                                        cropped_image.show()
                                    except Exception as e:
                                        print(f"Error with image: {e}")
                                print(f"Detected empty text crop: {rem_frac} - {debug}")
                            else:
                                multiple_crops.append(cropped_image)
                                if hasattr(element, "text"):
                                    print(f"\nOLD TEXT: {element.text}")
                    else:
                        print("Not a text element")
                if len(multiple_crops) > 0:
                    # good crops
                    return [PostOcrEnrichmentElement(item=element, image=multiple_crops)]
                else:
                    # nothing
                    return []

        @classmethod
        def get_options_type(cls) -> type[PictureDescriptionApiOptions]:
            return PictureDescriptionApiOptions

        def __init__(
            self,
            *,
            enabled: bool,
            enable_remote_services: bool,
            artifacts_path: Optional[Union[Path, str]],
            options: PictureDescriptionApiOptions,
            accelerator_options: AcceleratorOptions,
        ):
            self.enabled = enabled
            self.options = options
            self.concurrency = 2
            self.expansion_factor = 0.05
            self.table_cell_expansion_factor = 0.0  # do not modify table cell size
            self.elements_batch_size = 4
            self._accelerator_options = accelerator_options
            self._artifacts_path = (
                Path(artifacts_path) if isinstance(artifacts_path, str) else artifacts_path
            )

            if self.enabled and not enable_remote_services:
                raise OperationNotAllowed(
                    "Enable remote services by setting pipeline_options.enable_remote_services=True."
                )

        def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
            return self.enabled

        def _annotate_images(self, images: Iterable[Image.Image]) -> Iterable[str]:
            def _api_request(image: Image.Image) -> str:
                res = api_image_request(
                    image=image,
                    prompt=self.options.prompt,
                    url=self.options.url,
                    # timeout=self.options.timeout,
                    timeout=30,
                    headers=self.options.headers,
                    **self.options.params,
                )
                return res[0]

            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                yield from executor.map(_api_request, images)

        def __call__(
            self,
            doc: DoclingDocument,
            element_batch: Iterable[ItemAndImageEnrichmentElement],
        ) -> Iterable[NodeItem]:
            if not self.enabled:
                for element in element_batch:
                    yield element.item
                return

            elements: list[TextItem] = []
            images: list[Image.Image] = []
            img_ind_per_element: list[int] = []

            for element_stack in element_batch:
                for element in element_stack:
                    allowed = (DocItem, TableCell, RichTableCell, GraphCell)
                    assert isinstance(element.item, allowed)
                    for ind, img in enumerate(element.image):
                        elements.append(element.item)
                        images.append(img)
                        # images.append(element.image)
                        img_ind_per_element.append(ind)

            if not images:
                return

            outputs = list(self._annotate_images(images))

            for item, output, img_ind in zip(elements, outputs, img_ind_per_element):
                # Sometimes model can return html tags, which are not strictly needed in our, so it's better to clean them
                def clean_html_tags(text):
                    for tag in [
                        "<table>",
                        "<tr>",
                        "<td>",
                        "<strong>",
                        "</table>",
                        "</tr>",
                        "</td>",
                        "</strong>",
                        "<th>",
                        "</th>",
                        "<tbody>",
                        "<tbody>",
                        "<thead>",
                        "</thead>",
                    ]:
                        text = text.replace(tag, "")
                    return text

                output = clean_html_tags(output).strip()
                output = remove_break_lines(output)
                # The last measure against hallucinations
                # Detect hallucinated string...
                if output.startswith("The first of these"):
                    output = ""

                if no_long_repeats(output, 50):
                    if VERBOSE:
                        if isinstance(item, (TextItem)):
                            print(f"\nOLD TEXT: {item.text}")

                    # Re-populate text
                    if isinstance(item, TextItem | GraphCell):
                        if img_ind > 0:
                            # Concat texts across several provenances
                            item.text += " " + output
                            # item.orig += " " + output
                        else:
                            item.text = output
                            # item.orig = output
                    elif isinstance(item, TableCell | RichTableCell):
                        item.text = output
                    elif isinstance(item, PictureItem):
                        pass
                    else:
                        raise ValueError(f"Unknown item type: {type(item)}")

                    if VERBOSE:
                        if isinstance(item, (TextItem)):
                            print(f"NEW TEXT: {item.text}")

                    # Take care of charspans for relevant types
                    if isinstance(item, GraphCell):
                        item.prov.charspan = (0, len(item.text))
                    elif isinstance(item, TextItem):
                        item.prov[0].charspan = (0, len(item.text))

                yield item


    def convert_pdf(pdf_path: Path, out_intermediate_json: Path):
        # Let's prepare a Docling document json with embedded page images
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        # pipeline_options.images_scale = 4.0
        pipeline_options.images_scale = 2.0

        doc_converter = (
            DocumentConverter(  # all of the below is optional, has internal defaults.
                allowed_formats=[InputFormat.PDF],
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=StandardPdfPipeline, pipeline_options=pipeline_options
                    )
                },
            )
        )

        if VERBOSE:
            print(
                "Converting PDF to get a Docling document json with embedded page images..."
            )
        conv_result = doc_converter.convert(pdf_path)
        conv_result.document.save_as_json(
            filename=out_intermediate_json, image_mode=ImageRefMode.EMBEDDED
        )
        if PRINT_RESULT_MARKDOWN:
            md1 = conv_result.document.export_to_markdown()
            print("*** ORIGINAL MARKDOWN ***")
            print(md1)


    def post_process_json(in_json: Path, out_final_json: Path):
        # Post-Process OCR on top of existing Docling document, per element's bounding box:
        print(f"Post-process all bounding boxes with OCR... {os.path.basename(in_json)}")
        pipeline_options = PostOcrEnrichmentPipelineOptions(
            api_options=PictureDescriptionApiOptions(
                url=LM_STUDIO_URL,
                prompt=DEFAULT_PROMPT,
                provenance="lm-studio-ocr",
                batch_size=4,
                concurrency=2,
                scale=2.0,
                params={"model": LM_STUDIO_MODEL},
            )
        )

        doc_converter = DocumentConverter(
            format_options={
                InputFormat.JSON_DOCLING: FormatOption(
                    pipeline_cls=PostOcrEnrichmentPipeline,
                    pipeline_options=pipeline_options,
                    backend=DoclingJSONBackend,
                )
            }
        )
        result = doc_converter.convert(in_json)
        if SHOW_IMAGE:
            result.document.pages[1].image.pil_image.show()
        result.document.save_as_json(out_final_json)
        if PRINT_RESULT_MARKDOWN:
            md = result.document.export_to_markdown()
            print("*** MARKDOWN ***")
            print(md)


    def process_pdf(pdf_path: Path, scratch_dir: Path, out_dir: Path):
        inter_json = scratch_dir / (pdf_path.stem + ".json")
        final_json = out_dir / (pdf_path.stem + ".json")
        inter_json.parent.mkdir(parents=True, exist_ok=True)
        final_json.parent.mkdir(parents=True, exist_ok=True)
        if final_json.exists() and final_json.stat().st_size > 0:
            print(f"Result already found here: '{final_json}', aborting...")
            return  # already done
        convert_pdf(pdf_path, inter_json)
        post_process_json(inter_json, final_json)


    def process_json(json_path: Path, out_dir: Path):
        final_json = out_dir / (json_path.stem + ".json")
        final_json.parent.mkdir(parents=True, exist_ok=True)
        if final_json.exists() and final_json.stat().st_size > 0:
            return  # already done
        post_process_json(json_path, final_json)


    def filter_jsons_by_ocr_list(jsons, folder):
        """
        jsons: list[Path] - JSON files
        folder: Path - folder containing ocr_documents.txt
        """
        ocr_file = folder / "ocr_documents.txt"

        # If the file doesn't exist, return the list unchanged
        if not ocr_file.exists():
            return jsons

        # Read file names (strip whitespace, ignore empty lines)
        with ocr_file.open("r", encoding="utf-8") as f:
            allowed = {line.strip() for line in f if line.strip()}

        # Keep only JSONs whose stem is in allowed list
        filtered = [p for p in jsons if p.stem in allowed]
        return filtered


    def run_jsons(in_path: Path, out_dir: Path):
        if in_path.is_dir():
            jsons = sorted(in_path.glob("*.json"))
            if not jsons:
                raise SystemExit("Folder mode expects one or more .json files")
            # Look for ocr_documents.txt, in case found, respect only the jsons
            filtered_jsons = filter_jsons_by_ocr_list(jsons, in_path)
            for j in tqdm(filtered_jsons):
                print("")
                print("Processing file...")
                print(j)
                process_json(j, out_dir)
        else:
            raise SystemExit("Invalid --in path")


def main():
    logging.getLogger().setLevel(logging.ERROR)
    p = argparse.ArgumentParser(description="PDF/JSON -> final JSON pipeline")
    p.add_argument(
        "--in",
        dest="in_path",
        default="tests/data/pdf/2305.03393v1-pg9.pdf",
        required=False,
        help="Path to a PDF/JSON file or a folder of JSONs",
    )
    p.add_argument(
        "--out",
        dest="out_dir",
        default="scratch/",
        required=False,
        help="Folder for final JSONs (scratch goes inside)",
    )
    p.add_argument(
        "--fix-labels",
        action="store_true",
        help="Fix 'label' field in OCR analysis JSON using VLM on the region above each field bbox",
    )
    p.add_argument(
        "--ocr-analysis",
        dest="ocr_analysis",
        type=str,
        default=None,
        help="Path to OCR analysis JSON (from ocr_field_analysis.py). Required when using --fix-labels.",
    )
    p.add_argument(
        "--pdf",
        dest="pdf_path",
        type=str,
        default=None,
        help="Path to source PDF. Required when using --fix-labels (used to render page images).",
    )
    p.add_argument(
        "--apply-data",
        action="store_true",
        help="Apply applicant/data JSON to OCR analysis for given pages, then run generate_multi_analysis_overlays.",
    )
    p.add_argument(
        "--data",
        dest="data_path",
        type=str,
        default=None,
        help="Path to applicant/data JSON (e.g. data_clean_2.json). Required when using --apply-data.",
    )
    p.add_argument(
        "--pages",
        dest="pages",
        type=str,
        default="6,7,8,9",
        help="Comma-separated page numbers for --apply-data (default: 6,7,8,9).",
    )
    p.add_argument(
        "--mapping",
        dest="mapping_path",
        type=str,
        default=None,
        help="Optional JSON mapping file: { page_num: { field_id: data_path } }. Used with --apply-data.",
    )
    p.add_argument(
        "--no-prompt",
        action="store_true",
        dest="no_prompt",
        help="With --apply-data, do not prompt before applying; apply and run overlays immediately.",
    )
    p.add_argument(
        "--annotate-with-vlm",
        action="store_true",
        dest="annotate_with_vlm",
        help="Iteratively call LM Studio with form pages and data until all fields on given pages are annotated.",
    )
    p.add_argument(
        "--max-iterations",
        dest="max_iterations",
        type=int,
        default=3,
        help="Max VLM rounds per page for --annotate-with-vlm (default: 3).",
    )
    args = p.parse_args()

    # Annotate-with-VLM mode: iterative LM Studio calls with data + form pages until fields filled
    if args.annotate_with_vlm:
        data_path = args.data_path
        ocr_analysis_path = args.ocr_analysis
        pdf_path = args.pdf_path
        if not data_path or not ocr_analysis_path or not pdf_path:
            raise SystemExit(
                "With --annotate-with-vlm you must provide --data, --ocr-analysis, and --pdf. "
                "Example: --annotate-with-vlm --data path/to/data_clean_2.json --ocr-analysis path/to/ocr_analysis.json --pdf path/to/form.pdf --pages 6,7,8"
            )
        data_path = Path(data_path).expanduser().resolve()
        ocr_analysis_path = Path(ocr_analysis_path).expanduser().resolve()
        pdf_path = Path(pdf_path).expanduser().resolve()
        if not data_path.exists():
            raise SystemExit(f"Data JSON not found: {data_path}")
        if not ocr_analysis_path.exists():
            raise SystemExit(f"OCR analysis JSON not found: {ocr_analysis_path}")
        if not pdf_path.exists():
            raise SystemExit(f"PDF not found: {pdf_path}")
        try:
            pages = [int(x.strip()) for x in args.pages.split(",") if x.strip()]
        except ValueError:
            raise SystemExit("--pages must be comma-separated integers (e.g. 6,7,8)")
        run_annotate_pages_with_vlm_iterative(
            data_path=data_path,
            ocr_analysis_path=ocr_analysis_path,
            pdf_path=pdf_path,
            pages=pages,
            max_iterations=args.max_iterations,
        )
        return

    # Apply-data mode: merge data JSON into OCR analysis for pages 6-9, then run overlays
    if args.apply_data:
        data_path = args.data_path
        ocr_analysis_path = args.ocr_analysis
        pdf_path = args.pdf_path
        if not data_path or not ocr_analysis_path or not pdf_path:
            raise SystemExit(
                "With --apply-data you must provide --data, --ocr-analysis, and --pdf. "
                "Example: --apply-data --data /path/to/data_clean_2.json --ocr-analysis path/to/ocr_analysis.json --pdf path/to/form.pdf"
            )
        data_path = Path(data_path).expanduser().resolve()
        ocr_analysis_path = Path(ocr_analysis_path).expanduser().resolve()
        pdf_path = Path(pdf_path).expanduser().resolve()
        if not data_path.exists():
            raise SystemExit(f"Data JSON not found: {data_path}")
        if not ocr_analysis_path.exists():
            raise SystemExit(f"OCR analysis JSON not found: {ocr_analysis_path}")
        if not pdf_path.exists():
            raise SystemExit(f"PDF not found: {pdf_path}")
        try:
            pages = [int(x.strip()) for x in args.pages.split(",") if x.strip()]
        except ValueError:
            raise SystemExit("--pages must be comma-separated integers (e.g. 6,7,8,9)")
        mapping_path = Path(args.mapping_path).expanduser().resolve() if args.mapping_path else None
        run_apply_data_and_overlays(
            data_path=data_path,
            ocr_analysis_path=ocr_analysis_path,
            pdf_path=pdf_path,
            pages=pages,
            mapping_path=mapping_path,
            prompt_before_apply=not args.no_prompt,
        )
        return

    # Fix-labels mode: update labels in OCR analysis JSON using VLM
    if args.fix_labels:
        pdf_path = args.pdf_path
        ocr_analysis_path = args.ocr_analysis
        if not ocr_analysis_path or not pdf_path:
            raise SystemExit(
                "With --fix-labels you must provide --ocr-analysis and --pdf. "
                "Example: --fix-labels --pdf path/to/form.pdf --ocr-analysis path/to/form_ocr_analysis.json"
            )
        pdf_path = Path(pdf_path).expanduser().resolve()
        ocr_analysis_path = Path(ocr_analysis_path).expanduser().resolve()
        if not pdf_path.exists():
            raise SystemExit(f"PDF not found: {pdf_path}")
        if not ocr_analysis_path.exists():
            raise SystemExit(f"OCR analysis JSON not found: {ocr_analysis_path}")
        fix_ocr_analysis_labels(pdf_path, ocr_analysis_path)
        return

    in_path = Path(args.in_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    print(f"in_path: {in_path}")
    print(f"out_dir: {out_dir}")
    scratch_dir = out_dir / "temp"

    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    if in_path.is_file():
        if in_path.suffix.lower() == ".pdf":
            process_pdf(in_path, scratch_dir, out_dir)
        elif in_path.suffix.lower() == ".json":
            process_json(in_path, out_dir)
        else:
            raise SystemExit("Single-file mode expects a .pdf or .json")
    else:
        run_jsons(in_path, out_dir)


if __name__ == "__main__":
    main()
