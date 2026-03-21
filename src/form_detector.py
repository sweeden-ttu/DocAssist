import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from PIL import Image
import fitz


class FormDetector:
    FIELD_TYPES = [
        "text_input",
        "checkbox",
        "radio_button",
        "signature",
        "date",
        "currency",
        "ssn",
        "phone",
        "address",
        "name",
    ]

    def __init__(self, lmstudio_url: str = "http://localhost:1234"):
        from lmstudio_client import LMStudioClient

        self.client = LMStudioClient(lmstudio_url)
        if not self.client.is_available():
            raise RuntimeError("LM Studio is not available. Please start the server.")
        self.detected_fields: List[Dict[str, Any]] = []

    def pdf_to_images(
        self, pdf_path: str, output_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = pdf_path.parent / f"{pdf_path.stem}_images"
            output_dir.mkdir(parents=True, exist_ok=True)

        images: List[Dict[str, Any]] = []
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=200)
            img_path = output_dir / f"page_{page_num + 1:03d}.png"
            pix.save(str(img_path))
            images.append({"page": page_num + 1, "path": str(img_path)})

        doc.close()
        return images

    def detect_fields(
        self, image_path: str, form_type: str = "IRS Form", page: int = 1
    ) -> Dict[str, Any]:
        prompt = self._build_detection_prompt(form_type, page)
        result = self.client.extract_form_fields(image_path, prompt)
        result["image_path"] = image_path
        result["extracted_at"] = datetime.utcnow().isoformat() + "Z"
        self.detected_fields.append(result)
        return result

    def _build_detection_prompt(self, form_type: str, page: int) -> str:
        field_types_str = ", ".join(self.FIELD_TYPES)

        return f"""You are an expert at analyzing {form_type} documents.
Examine the form image carefully and identify ALL fillable fields.

For each field, determine:
1. **type**: One of [{field_types_str}]
2. **label**: Clear description of what the field is for
3. **bbox_2d**: Bounding box as [x1, y1, x2, y2] pixel coordinates
4. **confidence**: Your confidence score (0.0-1.0)
5. **fillable**: Whether this is an input field (usually true)

Be thorough - check for:
- Text input boxes (wages, deductions, etc.)
- Checkboxes (filing status, credits, etc.)
- Currency fields (amounts in dollars)
- Date fields
- Signature lines
- SSN fields
- Address fields

Return ONLY valid JSON with this structure:
{{
  "form_type": "{form_type}",
  "page": {page},
  "fields": [
    {{
      "id": "field_001",
      "type": "text_input",
      "label": "Your response label here",
      "bbox_2d": [100, 50, 300, 75],
      "confidence": 0.95,
      "fillable": true
    }}
  ]
}}"""

    def process_pdf(
        self, pdf_path: str, output_dir: str, form_type: str = "IRS Form"
    ) -> List[Dict[str, Any]]:
        images = self.pdf_to_images(pdf_path, output_dir)
        results: List[Dict[str, Any]] = []

        for img_info in images:
            print(f"Processing page {img_info['page']}...")
            result = self.detect_fields(img_info["path"], form_type, img_info["page"])
            results.append(result)

        return results


def save_results(results: List[Dict[str, Any]], output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")


def create_field_summary(results: List[Dict[str, Any]], output_path: str):
    summary: Dict[str, Any] = {
        "total_pages": len(results),
        "total_fields": sum(len(r.get("fields", [])) for r in results),
        "field_types": {},
        "fields": [],
    }

    for page_result in results:
        for field in page_result.get("fields", []):
            field_type = field.get("type", "unknown")
            summary["field_types"][field_type] = (
                summary["field_types"].get(field_type, 0) + 1
            )
            field["page"] = page_result.get("page", 1)
            summary["fields"].append(field)

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract form fields from IRS forms")
    parser.add_argument("--input", "-i", required=True, help="Input PDF or image file")
    parser.add_argument(
        "--output", "-o", default="output/form_fields.json", help="Output JSON file"
    )
    parser.add_argument(
        "--summary", "-s", default="output/field_summary.json", help="Summary JSON file"
    )
    parser.add_argument("--form-type", default="IRS Form", help="Form type for context")
    parser.add_argument(
        "--lmstudio-url", default="http://localhost:1234", help="LM Studio URL"
    )

    args = parser.parse_args()

    detector = FormDetector(args.lmstudio_url)
    input_path = Path(args.input)

    if input_path.suffix.lower() == ".pdf":
        results = detector.process_pdf(
            str(input_path), str(input_path.parent / "images"), args.form_type
        )
    else:
        results = [detector.detect_fields(str(input_path), args.form_type)]

    save_results(results, args.output)
    create_field_summary(results, args.summary)

    print(f"\nExtraction complete!")
    print(f"  Pages processed: {len(results)}")
    print(f"  Total fields: {sum(len(r.get('fields', [])) for r in results)}")


if __name__ == "__main__":
    main()
