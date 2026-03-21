import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from PIL import Image
import fitz


def setup_directories(base_dir: str = "output") -> Dict[str, Path]:
    dirs = {
        "base": Path(base_dir),
        "images": Path(base_dir) / "images",
        "detections": Path(base_dir) / "detections",
        "visualizations": Path(base_dir) / "visualizations",
        "training": Path(base_dir) / "training",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def preprocess_image(
    image_path: str,
    output_path: Optional[str] = None,
    dpi: int = 200,
    deskew: bool = True,
) -> str:
    img = Image.open(image_path)

    if deskew:
        from PIL import ImageOps

        img = ImageOps.autocontrast(img)

    if output_path is None:
        output_path = image_path.replace(".png", "_processed.png")

    img.save(output_path, dpi=(dpi, dpi))
    return output_path


def extract_page_dimensions(image_path: str) -> Dict[str, int]:
    img = Image.open(image_path)
    return {"width": img.width, "height": img.height, "mode": img.mode}


def pdf_to_images_batch(
    pdf_path: str, output_dir: str, dpi: int = 200
) -> List[Dict[str, Any]]:
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)
        img_path = output_dir / f"page_{page_num + 1:03d}.png"
        pix.save(str(img_path))

        dims = extract_page_dimensions(str(img_path))
        images.append({"page": page_num + 1, "path": str(img_path), **dims})

    doc.close()
    return images


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def normalize_bbox(bbox: List[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = bbox
    return [x1 / width, y1 / height, x2 / width, y2 / height]


def denormalize_bbox(bbox: List[float], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = bbox
    return [int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)]


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    import yaml

    config_path = Path(config_path)
    if not config_path.exists():
        return get_default_config()

    with open(config_path) as f:
        return yaml.safe_load(f)


def get_default_config() -> Dict[str, Any]:
    return {
        "lmstudio_url": "http://localhost:1234",
        "model_name": "lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF",
        "detection": {
            "confidence_threshold": 0.7,
            "iou_threshold": 0.5,
            "max_detections": 100,
        },
        "preprocessing": {"dpi": 200, "deskew": True, "binarize": False},
        "output": {"format": "standard", "save_images": False},
    }


def validate_detection_result(result: Dict[str, Any]) -> bool:
    required_keys = ["form_type", "page", "fields"]
    if not all(key in result for key in required_keys):
        return False

    for field in result["fields"]:
        if "bbox_2d" not in field:
            return False
        if len(field["bbox_2d"]) != 4:
            return False

    return True


def merge_overlapping_detections(
    fields: List[Dict[str, Any]], iou_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    if not fields:
        return []

    merged = [fields[0]]

    for field in fields[1:]:
        is_overlap = False
        for m in merged:
            if calculate_iou(field["bbox_2d"], m["bbox_2d"]) > iou_threshold:
                if field.get("confidence", 0) > m.get("confidence", 0):
                    merged.remove(m)
                    merged.append(field)
                is_overlap = True
                break

        if not is_overlap:
            merged.append(field)

    return merged


def filter_by_confidence(
    fields: List[Dict[str, Any]], threshold: float = 0.7
) -> List[Dict[str, Any]]:
    return [f for f in fields if f.get("confidence", 0) >= threshold]


def format_field_for_json(
    field: Dict[str, Any], image_width: int, image_height: int
) -> Dict[str, Any]:
    formatted = {
        "id": field.get("id", ""),
        "type": field.get("type", "text_input"),
        "label": field.get("label", ""),
        "bbox_2d": [int(x) for x in field.get("bbox_2d", [])],
        "confidence": float(field.get("confidence", 1.0)),
        "fillable": bool(field.get("fillable", True)),
    }

    if formatted["bbox_2d"]:
        formatted["bbox_normalized"] = normalize_bbox(
            formatted["bbox_2d"], image_width, image_height
        )

    return formatted
