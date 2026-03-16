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
from typing import Dict, Iterable, List, Tuple

import pypdfium2 as pdfium
from PIL import Image, ImageDraw


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


def extract_boxes_generic(path: Path) -> Dict[int, List[Tuple[float, float, float, float]]]:
    """Dispatch extraction based on the known file pattern."""
    data = load_json(path)

    if "form_fields" in data:
        # Coordinates-style data
        return extract_boxes_from_coordinates(data)

    # Default: per-page `fields` structure
    return extract_boxes_from_page_fields(data)


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


def create_overlay_for_page(
    page,
    page_num: int,
    page_height: float,
    analyses: Iterable[AnalysisConfig],
    boxes_by_analysis: Dict[str, Dict[int, List[Tuple[float, float, float, float]]]],
    scale: float = 2.0,
) -> Image.Image:
    """Create a transparent overlay image for a given page."""
    width = int(page.get_width() * scale)
    height = int(page.get_height() * scale)

    overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    for analysis in analyses:
        page_boxes = boxes_by_analysis.get(analysis.name, {}).get(page_num, [])
        if not page_boxes:
            continue

        for x0, y0, x1, y1 in page_boxes:
            # Coordinates are in PDF space with origin at bottom-left.
            # Convert to image coordinates (origin at top-left).
            x0_s = x0 * scale
            x1_s = x1 * scale

            # y0/y1 measured from bottom; convert to top/bottom in image space.
            y0_img = (page_height - y0) * scale
            y1_img = (page_height - y1) * scale

            top = min(y0_img, y1_img)
            bottom = max(y0_img, y1_img)

            draw.rectangle(
                [x0_s, top, x1_s, bottom],
                outline=analysis.color,
                width=2,
            )

    return overlay


def generate_multi_analysis_overlays(
    pdf_path: Path,
    analyses: Iterable[AnalysisConfig],
    overlays_dir: Path,
    annotated_dir: Path,
    scale: float = 2.0,
) -> None:
    overlays_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    pdf = pdfium.PdfDocument(str(pdf_path))
    num_pages = len(pdf)

    boxes_by_analysis = build_all_boxes(analyses)

    # Assume all analyses share the same page height as the PDF.
    first_page = pdf.get_page(0)
    page_height = float(first_page.get_height())

    for page_index in range(num_pages):
        page_num = page_index + 1
        page = pdf.get_page(page_index)

        # Create overlay for this page across all analyses.
        overlay = create_overlay_for_page(
            page=page,
            page_num=page_num,
            page_height=page_height,
            analyses=analyses,
            boxes_by_analysis=boxes_by_analysis,
            scale=scale,
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

    overlays_dir = annotations_dir / "overlays"
    annotated_dir = annotations_dir / "annotated"

    generate_multi_analysis_overlays(
        pdf_path=pdf_path,
        analyses=analyses,
        overlays_dir=overlays_dir,
        annotated_dir=annotated_dir,
        scale=2.0,
    )

