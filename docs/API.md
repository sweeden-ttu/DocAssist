# DocAssist API Documentation

## Overview
DocAssist provides a Python API for extracting form fields from IRS documents using vision-language models via LM Studio.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.form_detector import FormDetector
from src.field_extractor import FieldExtractor
from src.json_converter import JSONConverter

# Initialize
detector = FormDetector("http://localhost:1234")

# Extract fields from image
result = detector.detect_fields("form_image.png", "IRS Form 1040")

# Enhance with metadata
extractor = FieldExtractor()
enhanced = extractor.extract_with_context(result, "IRS Form 1040")

# Convert to different formats
converter = JSONConverter()
converter.convert_file("output.json", "coco_output.json", "coco")
```

## LMStudioClient

### Constructor
```python
LMStudioClient(base_url: str = "http://localhost:1234")
```

### Methods

#### `is_available() -> bool`
Check if LM Studio server is running.

#### `encode_image(image_path: str) -> str`
Encode image to base64 for API transmission.

#### `extract_form_fields(image_path: str, prompt: str = None) -> dict`
Extract form fields from an image using VLM.

**Parameters:**
- `image_path`: Path to the form image
- `prompt`: Custom prompt (optional)

**Returns:**
```json
{
  "form_type": "IRS Form 1040",
  "page": 1,
  "fields": [...]
}
```

## FormDetector

### Constructor
```python
FormDetector(lmstudio_url: str = "http://localhost:1234")
```

### Methods

#### `pdf_to_images(pdf_path: str, output_dir: str = None) -> List[Dict]`
Convert PDF to images for processing.

#### `detect_fields(image_path: str, form_type: str = "IRS Form", page: int = 1) -> dict`
Detect form fields in a single image.

#### `process_pdf(pdf_path: str, output_dir: str, form_type: str = "IRS Form") -> List[dict]`
Process entire PDF document.

## FieldExtractor

### Methods

#### `extract_with_context(detection: dict, form_type: str = "IRS Form 1040") -> dict`
Enhance detection results with IRS-specific context.

#### `generate_fill_template(detection: dict) -> dict`
Generate a fillable template from detection results.

#### `export_for_training(detections: List[dict], output_dir: str) -> List[TrainingExample]`
Export data for episodic training.

## JSONConverter

### Methods

#### `to_coco_format(detection: dict, image_width: int, image_height: int) -> dict`
Convert to COCO format for object detection.

#### `to_yolo_format(detection: dict, image_width: int, image_height: int) -> List[str]`
Convert to YOLO format.

#### `to_standard(detection: dict) -> dict`
Convert to standard DocAssist format.

## FieldVisualizer

### Methods

#### `draw_bounding_boxes(image_path: str, detection: dict, output_path: str)`
Draw bounding boxes on image.

## EpisodicTrainer

### Constructor
```python
EpisodicTrainer(n_way: int = 5, k_shot: int = 5, n_episodes: int = 100)
```

### Methods

#### `prepare_training_data(json_path: str) -> List[TrainingExample]`
Load training examples from JSON.

#### `generate_episodes(training_data: List[TrainingExample]) -> List[Episode]`
Generate episodic training episodes.

#### `export_episodes(episodes: List[Episode], output_dir: str)`
Export episodes for training.

#### `create_lora_config(output_path: str = "configs/lora_config.json")`
Generate LoRA configuration.

## Output Formats

### Standard Format
```json
{
  "form_type": "IRS Form 1040",
  "page": 1,
  "fields": [
    {
      "id": "field_001",
      "type": "text_input",
      "label": "First name",
      "bbox_2d": [x1, y1, x2, y2],
      "confidence": 0.95,
      "fillable": true
    }
  ]
}
```

### COCO Format
```json
{
  "images": [...],
  "annotations": [...],
  "categories": [...]
}
```

### YOLO Format
```
class_id x_center y_center width height
```

## CLI Usage

```bash
# Extract fields from PDF
python src/form_detector.py --input form.pdf --output output.json

# Convert format
python src/json_converter.py --input output.json --format coco --output coco.json

# Generate training episodes
python src/episodic_trainer.py --input training_data.json --output episodes/ --n-way 5 --k-shot 5
```
