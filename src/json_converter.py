import json
import os
from pathlib import Path
from typing import Optional, Literal
from PIL import Image, ImageDraw, ImageFont
import random


class JSONConverter:
    OUTPUT_FORMATS = ["coco", "pascal_voc", "yolo", "qwen", "custom"]

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def to_coco_format(
        self, detection: dict, image_width: int, image_height: int
    ) -> dict:
        coco_output = {
            "images": [
                {
                    "id": 1,
                    "file_name": Path(detection.get("image_path", "image.png")).name,
                    "width": image_width,
                    "height": image_height,
                }
            ],
            "annotations": [],
            "categories": self._get_coco_categories(),
        }

        annotation_id = 1
        for field in detection.get("fields", []):
            bbox = field.get("bbox_2d", [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox

            coco_bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

            coco_output["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": 1,
                    "category_id": self._get_category_id(
                        field.get("type", "text_input")
                    ),
                    "bbox": coco_bbox,
                    "area": coco_bbox[2] * coco_bbox[3],
                    "iscrowd": 0,
                    "attributes": {
                        "label": field.get("label", ""),
                        "confidence": field.get("confidence", 1.0),
                    },
                }
            )
            annotation_id += 1

        return coco_output

    def to_yolo_format(
        self, detection: dict, image_width: int, image_height: int
    ) -> tuple:
        yolo_lines = []

        for field in detection.get("fields", []):
            bbox = field.get("bbox_2d", [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox

            x_center = ((x1 + x2) / 2) / image_width
            y_center = ((y1 + y2) / 2) / image_height
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height

            class_id = self._get_category_id(field.get("type", "text_input")) - 1

            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        return yolo_lines

    def to_standard(self, detection: dict) -> dict:
        standard_output = {
            "form_type": detection.get("form_type", "Unknown"),
            "page": detection.get("page", 1),
            "image_path": detection.get("image_path"),
            "extracted_at": detection.get("extracted_at"),
            "fields": [],
        }

        for field in detection.get("fields", []):
            bbox = field.get("bbox_2d", [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox

            standard_field = {
                "id": field.get("id", ""),
                "type": field.get("type", "text_input"),
                "label": field.get("label", ""),
                "bbox_2d": [int(x1), int(y1), int(x2), int(y2)],
                "bbox_normalized": [x1 / 2000, y1 / 2500, x2 / 2000, y2 / 2500],
                "confidence": field.get("confidence", 1.0),
                "fillable": field.get("fillable", True),
            }
            standard_output["fields"].append(standard_field)

        return standard_output

    def convert_file(self, input_json: str, output_json: str, format: str = "standard"):
        with open(input_json) as f:
            detection = json.load(f)

        if format == "standard":
            result = self.to_standard(detection)
        elif format == "coco":
            result = self.to_coco_format(detection, 2000, 2500)
        elif format == "yolo":
            yolo_lines = self.to_yolo_format(detection, 2000, 2500)
            result = {"lines": yolo_lines}
        else:
            result = self.to_standard(detection)

        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def _get_coco_categories(self) -> list:
        return [
            {"id": 1, "name": "text_input", "supercategory": "field"},
            {"id": 2, "name": "checkbox", "supercategory": "field"},
            {"id": 3, "name": "radio_button", "supercategory": "field"},
            {"id": 4, "name": "signature", "supercategory": "field"},
            {"id": 5, "name": "date", "supercategory": "field"},
            {"id": 6, "name": "currency", "supercategory": "field"},
            {"id": 7, "name": "ssn", "supercategory": "field"},
        ]

    def _get_category_id(self, field_type: str) -> int:
        categories = {
            "text_input": 1,
            "checkbox": 2,
            "radio_button": 3,
            "signature": 4,
            "date": 5,
            "currency": 6,
            "ssn": 7,
            "phone": 8,
            "address": 9,
            "name": 10,
        }
        return categories.get(field_type, 1)


class FieldVisualizer:
    COLORS = {
        "text_input": (255, 0, 0),
        "checkbox": (0, 255, 0),
        "radio_button": (0, 0, 255),
        "signature": (255, 0, 255),
        "date": (255, 165, 0),
        "currency": (255, 255, 0),
        "ssn": (0, 255, 255),
        "default": (128, 128, 128),
    }

    def __init__(self):
        self.colors = self.COLORS

    def draw_bounding_boxes(self, image_path: str, detection: dict, output_path: str):
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        for field in detection.get("fields", []):
            bbox = field.get("bbox_2d", [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            color = self.colors.get(
                field.get("type", "default"), self.colors["default"]
            )

            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            label = f"{field.get('type', 'field')}: {field.get('label', '')[:20]}"
            draw.text((x1, y1 - 10), label, fill=color)

        img.save(output_path)
        print(f"Visualization saved to {output_path}")

    def create_field_map(
        self, detection: dict, image_width: int, image_height: int
    ) -> Image.Image:
        map_img = Image.new("RGB", (image_width, image_height), color="white")
        draw = ImageDraw.Draw(map_img)

        for field in detection.get("fields", []):
            bbox = field.get("bbox_2d", [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            color = self.colors.get(
                field.get("type", "default"), self.colors["default"]
            )

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            field_id = field.get("id", "?")
            draw.text((x1 + 5, y1 + 5), field_id, fill="black")

        return map_img


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert form detection results")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file")
    parser.add_argument(
        "--output", "-o", default="output/converted.json", help="Output JSON file"
    )
    parser.add_argument(
        "--format", "-f", choices=["standard", "coco", "yolo"], default="standard"
    )
    parser.add_argument("--visualize", "-v", help="Visualize bounding boxes on image")
    parser.add_argument("--image", help="Image file for visualization")

    args = parser.parse_args()

    converter = JSONConverter()
    converter.convert_file(args.input, args.output, args.format)

    if args.visualize and args.image:
        with open(args.input) as f:
            detection = json.load(f)

        visualizer = FieldVisualizer()
        visualizer.draw_bounding_boxes(args.image, detection, args.visualize)


if __name__ == "__main__":
    main()
