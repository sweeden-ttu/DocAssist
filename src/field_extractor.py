import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse


class FieldExtractor:
    IRS_FIELD_CONTEXT = {
        "text_input": {
            "description": "Free-text input fields for names, addresses, and general information",
            "examples": [
                "First name",
                "Last name",
                "Street address",
                "City",
                "State",
                "Occupation",
            ],
        },
        "checkbox": {
            "description": "Checkbox fields for binary choices like filing status or consent",
            "examples": [
                "Single",
                "Married filing jointly",
                "Head of household",
                "Qualifying widow(er)",
            ],
        },
        "currency": {
            "description": "Money/amount fields showing dollar values",
            "examples": [
                "Wages",
                "Salaries",
                "Total income",
                "Adjusted gross income",
                "Tax liability",
            ],
        },
        "date": {
            "description": "Date fields for dates of birth, filing, etc.",
            "examples": ["Date of birth", "Tax year", "Date signed"],
        },
        "signature": {
            "description": "Signature lines and blocks",
            "examples": ["Your signature", "Spouse's signature", "Preparer signature"],
        },
        "ssn": {
            "description": "Social Security Number fields",
            "examples": ["Your SSN", "Spouse's SSN", "Dependent SSN"],
        },
    }

    def __init__(self):
        self.extracted_fields: List[Dict[str, Any]] = []

    def extract_with_context(
        self, detection: Dict[str, Any], form_type: str = "IRS Form 1040"
    ) -> Dict[str, Any]:
        enhanced_result = detection.copy()
        enhanced_fields = []

        for field in detection.get("fields", []):
            enhanced_field = self._enhance_field(field, form_type)
            enhanced_fields.append(enhanced_field)

        enhanced_result["fields"] = enhanced_fields
        enhanced_result["form_type"] = form_type
        enhanced_result["metadata"] = {
            "extraction_version": "1.0",
            "context": "IRS Tax Form Field Extraction",
        }

        self.extracted_fields.extend(enhanced_fields)
        return enhanced_result

    def _enhance_field(self, field: Dict[str, Any], form_type: str) -> Dict[str, Any]:
        field_type = field.get("type", "text_input")

        enhanced = field.copy()

        enhanced["metadata"] = {
            "context": self.IRS_FIELD_CONTEXT.get(field_type, {}).get(
                "description", ""
            ),
            "examples": self.IRS_FIELD_CONTEXT.get(field_type, {}).get("examples", []),
            "is_likely_required": self._is_required_field(field),
            "field_category": self._categorize_field(field),
        }

        enhanced["validation"] = self._get_validation_rules(field_type)

        return enhanced

    def _is_required_field(self, field: Dict[str, Any]) -> bool:
        label = field.get("label", "").lower()
        required_keywords = [
            "your",
            "first name",
            "last name",
            "ssn",
            "social security",
        ]
        return any(kw in label for kw in required_keywords)

    def _categorize_field(self, field: Dict[str, Any]) -> str:
        field_type = field.get("type", "")

        if field_type in ["currency"]:
            return "financial"
        elif field_type in ["ssn"]:
            return "identification"
        elif field_type in ["signature"]:
            return "attestation"
        elif field_type in ["date"]:
            return "temporal"
        elif field_type in ["checkbox", "radio_button"]:
            return "selection"
        else:
            return "personal"

    def _get_validation_rules(self, field_type: str) -> Dict[str, Any]:
        rules = {
            "text_input": {"max_length": 50, "allowed_chars": "alphanumeric"},
            "checkbox": {"max_selections": 1, "required": False},
            "currency": {"format": "dollar_amount", "precision": 2},
            "date": {"format": "MM/DD/YYYY"},
            "signature": {"required": True, "type": "signature"},
            "ssn": {"format": "XXX-XX-XXXX", "pattern": r"^\d{3}-\d{2}-\d{4}$"},
        }
        return rules.get(field_type, {})

    def generate_fill_template(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        template = {
            "form_id": detection.get("form_type", "Unknown Form"),
            "version": "1.0",
            "fields": [],
        }

        for field in detection.get("fields", []):
            if field.get("fillable", True):
                template["fields"].append(
                    {
                        "id": field.get("id", ""),
                        "type": field.get("type", "text_input"),
                        "label": field.get("label", ""),
                        "required": field.get("metadata", {}).get(
                            "is_likely_required", False
                        ),
                        "value": None,
                        "bbox": field.get("bbox_2d", []),
                    }
                )

        return template

    def export_for_training(self, detections: List[Dict[str, Any]], output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        training_data = []

        for detection in detections:
            for field in detection.get("fields", []):
                training_item = {
                    "image_path": detection.get("image_path"),
                    "page": detection.get("page", 1),
                    "field_type": field.get("type"),
                    "label": field.get("label"),
                    "bbox_2d": field.get("bbox_2d"),
                    "category": field.get("metadata", {}).get(
                        "field_category", "unknown"
                    ),
                }
                training_data.append(training_item)

        output_file = output_path / "training_data.json"
        with open(output_file, "w") as f:
            json.dump(training_data, f, indent=2)

        print(f"Training data exported to {output_file}")
        return training_data


def main():
    parser = argparse.ArgumentParser(description="Extract and enhance form fields")
    parser.add_argument(
        "--input", "-i", required=True, help="Input JSON from form_detector.py"
    )
    parser.add_argument(
        "--output", "-o", default="output/enhanced_fields.json", help="Output JSON"
    )
    parser.add_argument("--template", "-t", help="Generate fill template JSON")
    parser.add_argument("--export-training", "-e", help="Export for episodic training")

    args = parser.parse_args()

    extractor = FieldExtractor()

    with open(args.input) as f:
        detection = json.load(f)

    if isinstance(detection, list):
        detection = detection[0] if detection else {}

    enhanced = extractor.extract_with_context(detection)

    with open(args.output, "w") as f:
        json.dump(enhanced, f, indent=2)

    print(f"Enhanced fields saved to {args.output}")

    if args.template:
        template = extractor.generate_fill_template(enhanced)
        with open(args.template, "w") as f:
            json.dump(template, f, indent=2)
        print(f"Fill template saved to {args.template}")

    if args.export_training:
        extractor.export_for_training([enhanced], args.export_training)


if __name__ == "__main__":
    main()
