# Quick Start Guide - DocAssist

## Prerequisites
1. LM Studio installed and running on port 1234
2. Python 3.10+
3. Download Qwen2.5-VL model in LM Studio

## Installation

```bash
cd /home/sweeden/projects/DocAssist
pip install -r requirements.txt
```

## Basic Usage

### 1. Extract fields from a form image
```bash
python run.py extract --input path/to/form.png --output output/fields.json
```

### 2. Process an entire PDF
```bash
python run.py extract --input path/to/form.pdf --form-type "IRS Form 1040"
```

### 3. Convert to different formats
```bash
python run.py convert --input output/fields.json --format coco --output output/coco.json
```

### 4. Visualize bounding boxes
```bash
python run.py visualize --input output/fields.json --image path/to/form.png --output output/visualization.png
```

### 5. Generate training episodes
```bash
python run.py train --input examples/sample_output.json --output output/episodes --n-way 5 --k-shot 5
```

## Python API

```python
from src.form_detector import FormDetector

detector = FormDetector("http://localhost:1234")
result = detector.detect_fields("form.png", "IRS Form 1040")
print(result)
```

## Example Output

```json
{
  "form_type": "IRS Form 1040",
  "page": 1,
  "fields": [
    {
      "id": "f1040_001",
      "type": "text_input",
      "label": "First name",
      "bbox_2d": [145, 52, 340, 72],
      "confidence": 0.95,
      "fillable": true
    }
  ]
}
```

## Next Steps
- See [docs/API.md](docs/API.md) for API reference
- See [docs/TRAINING.md](docs/TRAINING.md) for episodic training
- See [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for project roadmap
