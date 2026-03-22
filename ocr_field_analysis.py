#!/usr/bin/env python3
"""
OCR-based field analysis for USDA PDF forms using docling.
Renders each page as image, extracts text with OCR, and maps form fields to labels.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from PIL import Image
import pypdfium2 as pdfium
import pypdf


def render_page_to_image(
    pdf_path: str, page_num: int, scale: float = 2.0
) -> Image.Image:
    """Render a PDF page to an image."""
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf.get_page(page_num - 1)  # 0-indexed
    width = int(page.get_width() * scale)
    height = int(page.get_height() * scale)
    bitmap = page.render(scale=scale)
    img = bitmap.to_pil()
    # Resize if needed
    if img.width != width or img.height != height:
        img = img.resize((width, height), Image.LANCZOS)
    return img


def get_field_coordinates(pdf_path: str) -> dict:
    """Get form field coordinates from PDF."""
    reader = pypdf.PdfReader(pdf_path)
    fields = {}

    for page_num, page in enumerate(reader.pages, 1):
        if "/Annots" not in page:
            continue
        for annot in page["/Annots"]:
            annot_obj = annot.get_object()
            if annot_obj.get("/FT") in ["/Tx", "/Btn", "/Ch"]:  # Text, Button, Choice
                field_name = annot_obj.get("/T")
                if field_name:
                    field_name = str(field_name).strip()
                    rect = annot_obj.get("/Rect", [0, 0, 0, 0])
                    if len(rect) == 4:
                        fields[field_name] = {
                            "page": page_num,
                            "x": rect[0],
                            "y": rect[1],
                            "width": rect[2] - rect[0],
                            "height": rect[3] - rect[1],
                            "type": str(annot_obj.get("/FT", "/Tx")),
                        }
    return fields


def find_label_near_field(
    field_x: float,
    field_y: float,
    page_width: float,
    page_height: float,
    ocr_data: list,
    max_distance: float = 150,
) -> str:
    """Find the nearest text label to a field based on position.

    PDF coordinates: origin at bottom-left, y increases upward
    Image coordinates: origin at top-left, y increases downward
    """
    best_label = None
    best_distance = float("inf")

    for item in ocr_data:
        # OCR gives top-left origin, convert to PDF-like coordinates
        text_x = item["x"] + item["width"] / 2
        text_y = page_height - (item["y"] + item["height"] / 2)  # Flip Y

        # Prefer text to the left of the field
        if text_x >= field_x:
            continue

        # Calculate distance
        distance = ((field_x - text_x) ** 2 + (field_y - text_y) ** 2) ** 0.5

        if distance < best_distance and distance < max_distance:
            best_distance = distance
            best_label = item["text"]

    return best_label if best_label else ""


def find_label_above_field(
    field_x: float,
    field_y: float,
    page_width: float,
    page_height: float,
    ocr_data: list,
    max_distance: float = 100,
) -> str:
    """Find text above the field that could be a label."""
    best_label = None
    best_y = float("inf")

    for item in ocr_data:
        text_x = item["x"] + item["width"] / 2
        text_y = page_height - (item["y"] + item["height"] / 2)

        # Text should be above the field (higher y in PDF coords)
        if text_y <= field_y:
            continue

        # Should be roughly aligned horizontally
        if abs(text_x - field_x) > 50:
            continue

        # Find the one closest above
        y_diff = text_y - field_y
        if y_diff < best_y and y_diff < max_distance:
            best_y = y_diff
            best_label = item["text"]

    return best_label if best_label else ""


def analyze_pdf_with_ocr(
    pdf_path: str, output_dir: str, pages: Optional[set[int]] = None
):
    """Analyze PDF pages with OCR and map fields to labels."""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing: {pdf_path.name}")

    # Get page dimensions
    pdf = pdfium.PdfDocument(str(pdf_path))
    page_sizes = {}
    for i in range(len(pdf)):
        page = pdf.get_page(i)
        page_sizes[i + 1] = (page.get_width(), page.get_height())

    # Get field coordinates
    print("Extracting form fields...")
    fields = get_field_coordinates(str(pdf_path))
    print(f"Found {len(fields)} form fields")
    if not fields:
        print("No fillable fields found in PDF annotations.")

    # Group fields by page
    fields_by_page = {}
    for field_name, field_data in fields.items():
        page = field_data["page"]
        if page not in fields_by_page:
            fields_by_page[page] = []
        fields_by_page[page].append((field_name, field_data))

    # Process each page that has fields
    all_results = {}

    for page_num in sorted(fields_by_page.keys()):
        if pages and page_num not in pages:
            continue
        print(f"\n--- Page {page_num} ---")

        page_width, page_height = page_sizes.get(page_num, (612, 792))

        # Render page to image
        print(f"  Rendering page {page_num}...")
        image = render_page_to_image(str(pdf_path), page_num)

        # Save image for debugging
        image.save(output_dir / f"page_{page_num}.png")

        # Use OCR via pytesseract
        print(f"  Running OCR...")
        try:
            import pytesseract
        except ImportError as exc:
            raise SystemExit(
                "pytesseract is required for OCR. Install it with: pip install pytesseract"
            ) from exc

        custom_config = r"--oem 3 --psm 6"
        ocr_result = pytesseract.image_to_data(
            image, config=custom_config, output_type=pytesseract.Output.DICT
        )

        ocr_data = []
        for i in range(len(ocr_result["text"])):
            text = ocr_result["text"][i].strip()
            if text:
                ocr_data.append(
                    {
                        "text": text,
                        "x": ocr_result["left"][i],
                        "y": ocr_result["top"][i],
                        "width": ocr_result["width"][i],
                        "height": ocr_result["height"][i],
                        "confidence": ocr_result["conf"][i],
                    }
                )

        print(f"  Found {len(ocr_data)} text elements")

        # Map fields to labels
        page_results = []
        for field_name, field_data in fields_by_page[page_num]:
            # Field coordinates are already in PDF points (bottom-left origin)
            field_x = field_data["x"] + field_data["width"] / 2
            field_y = field_data["y"] + field_data["height"] / 2

            # Try to find label to the left
            label = find_label_near_field(
                field_x, field_y, page_width, page_height, ocr_data
            )

            # If no label to the left, try above
            if not label:
                label = find_label_above_field(
                    field_x, field_y, page_width, page_height, ocr_data
                )

            page_results.append(
                {
                    "field": field_name,
                    "label": label,
                    "field_coords": field_data,
                    "confidence": "high" if label else "unknown",
                }
            )

        all_results[f"page_{page_num}"] = {
            "page_size": {"width": page_width, "height": page_height},
            "field_count": len(page_results),
            "fields": page_results,
        }

        print(f"  Mapped {len(page_results)} fields to labels")

    # Save results
    output_file = output_dir / f"{pdf_path.stem}_ocr_analysis.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to: {output_file}")

    return all_results


def print_page_fields(pdf_path: str, page_num: int):
    """Print a single page with fields and their OCR-detected labels."""
    pdf_path = Path(pdf_path)

    print(f"\n{'=' * 80}")
    print(f"PAGE {page_num}")
    print(f"{'=' * 80}")

    # Get page dimensions
    pdf = pdfium.PdfDocument(str(pdf_path))
    page = pdf.get_page(page_num - 1)
    page_width = page.get_width()
    page_height = page.get_height()

    # Get fields for this page
    fields = get_field_coordinates(str(pdf_path))
    page_fields = {k: v for k, v in fields.items() if v["page"] == page_num}

    print(f"\nForm Fields ({len(page_fields)} total):")
    print("-" * 60)

    # Render page
    image = render_page_to_image(str(pdf_path), page_num)

    # OCR
    try:
        import pytesseract
    except ImportError as exc:
        raise SystemExit(
            "pytesseract is required for OCR. Install it with: pip install pytesseract"
        ) from exc

    custom_config = r"--oem 3 --psm 6"
    ocr_result = pytesseract.image_to_data(
        image, config=custom_config, output_type=pytesseract.Output.DICT
    )

    ocr_data = []
    for i in range(len(ocr_result["text"])):
        text = ocr_result["text"][i].strip()
        if text:
            ocr_data.append(
                {
                    "text": text,
                    "x": ocr_result["left"][i],
                    "y": ocr_result["top"][i],
                    "width": ocr_result["width"][i],
                    "height": ocr_result["height"][i],
                }
            )

    # Sort fields by Y position (top to bottom in PDF coords)
    sorted_fields = sorted(page_fields.items(), key=lambda x: x[1]["y"], reverse=True)

    for field_name, field_data in sorted_fields[:20]:  # Limit to first 20
        field_x = field_data["x"] + field_data["width"] / 2
        field_y = field_data["y"] + field_data["height"] / 2

        label = find_label_near_field(
            field_x, field_y, page_width, page_height, ocr_data
        )
        if not label:
            label = find_label_above_field(
                field_x, field_y, page_width, page_height, ocr_data
            )

        print(f"Field: {field_name}")
        print(f"  Coords: ({field_data['x']:.1f}, {field_data['y']:.1f})")
        print(f"  Label: {label if label else '(no label found)'}")
        print()


def _parse_page_list(raw: Optional[str]) -> Optional[set[int]]:
    if not raw:
        return None
    pages = set()
    for chunk in raw.split(","):
        value = chunk.strip()
        if not value:
            continue
        try:
            page_num = int(value)
        except ValueError as exc:
            raise SystemExit(
                f"Invalid page number '{value}'. Use comma-separated integers, e.g. 1,2,3"
            ) from exc
        if page_num < 1:
            raise SystemExit("Page numbers must be >= 1.")
        pages.add(page_num)
    return pages or None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract fillable PDF fields and OCR labels into *_ocr_analysis.json."
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to input PDF (for example: templates/irs/f1040.pdf)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write rendered pages and *_ocr_analysis.json",
    )
    parser.add_argument(
        "--pages",
        default=None,
        help="Optional comma-separated pages to process (1-based), e.g. 1,2",
    )
    parser.add_argument(
        "--print-pages",
        default=None,
        help="Optional comma-separated pages to print a field/label preview for debugging",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    pages = _parse_page_list(args.pages)
    print_pages = _parse_page_list(args.print_pages)

    analyze_pdf_with_ocr(str(pdf_path), args.output_dir, pages=pages)
    if print_pages:
        for page in sorted(print_pages):
            print_page_fields(str(pdf_path), page)
