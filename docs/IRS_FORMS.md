# IRS Form Documentation

## Supported Forms

### Form 1040 - U.S. Individual Income Tax Return
**Status**: In Development

### Field Types
| Field Type | Description | BBox Color |
|------------|-------------|------------|
| Text Input | Personal info, income fields | Blue |
| Checkbox | Filing status, credits | Green |
| Currency | Money amounts | Yellow |
| SSN | Social Security numbers | Red |
| Date | Date fields | Orange |
| Signature | Signature areas | Purple |

### Common Fields on Form 1040
- Filing status checkboxes (Single, Married, etc.)
- Name and address fields
- SSN fields
- Income fields ( wages, dividends, etc.)
- Deduction fields
- Signature block
- Date field
- Occupation field

## Output Format

```json
{
  "form_type": "IRS Form 1040",
  "form_version": "2024",
  "page": 1,
  "image_size": [816, 1008],
  "fields": [
    {
      "id": "f1040_field_001",
      "type": "text_input",
      "label": "First name",
      "bbox_2d": [145, 52, 340, 72],
      "bbox_normalized": [0.177, 0.052, 0.417, 0.071],
      "confidence": 0.95,
      "fillable": true
    },
    {
      "id": "f1040_field_002",
      "type": "checkbox",
      "label": "Single filing status",
      "bbox_2d": [120, 95, 135, 110],
      "bbox_normalized": [0.147, 0.094, 0.165, 0.109],
      "confidence": 0.92,
      "fillable": true
    }
  ],
  "metadata": {
    "extracted_at": "2024-01-15T10:30:00Z",
    "model_version": "qwen2.5-vl-7b-v1",
    "preprocessing": "deskew=True, binarize=True"
  }
}
```

## Processing Pipeline

```
1. PDF Input
   └─▶ PyMuPDF/RGBA conversion
        └─▶ Image preprocessing (optional)
             └─▶ VLM inference
                  └─▶ JSON output with bounding boxes
```

## Bounding Box Format

All coordinates are in pixels unless specified as normalized.

### COCO Format (Normalized)
```
[x_center, y_center, width, height] - normalized to [0, 1]
```

### Pascal VOC Format (Pixels)
```
[x_min, y_min, x_max, y_max]
```

### Qwen2.5-VL Format
```
bbox_2d: [x1, y1, x2, y2] - pixel coordinates
```

## Quality Metrics

- **Precision**: % of detected fields that are correct
- **Recall**: % of actual fields that are detected
- **mAP**: Mean Average Precision for bounding boxes
- **IoU**: Intersection over Union for localization

## Target Accuracy
- Field Detection: >95% recall
- Classification: >90% accuracy
- Localization (IoU): >0.8 average
