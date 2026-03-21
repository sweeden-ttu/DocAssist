# GUI Viewer Usage Guide

## Overview

The DocAssist GUI provides an interactive way to view form field detections with bounding box overlays.

## Installation

```bash
pip install PyQt5
# or
pip install PyQt6
```

## Launching the GUI

### Basic Launch
```bash
python src/gui_viewer.py
```

### With Files
```bash
python src/gui_viewer.py --image form.png --json fields.json
```

### From CLI Tool
```bash
python docling_cli.py gui --image form.png --json fields.json
```

## Features

### Image Display
- Pan and zoom controls
- Mouse wheel zoom
- Drag to pan
- Reset view button

### Bounding Box Display
- Color-coded by field type
- Transparency overlay
- Labels with confidence scores
- Toggle visibility

### Field Type Colors
| Type | Color |
|------|-------|
| text_input | Blue |
| checkbox | Green |
| radio_button | Light Green |
| signature | Purple |
| date | Orange |
| currency | Yellow |
| ssn | Cyan |
| phone | Pink |
| address | Violet |
| name | Teal |

### Interactive Table
- Click row to highlight field
- Sort by any column
- Filter by field type
- Copy field data

### Zoom Controls
- Mouse wheel zoom
- Slider control
- Fit to window
- Zoom percentage display

### Export Options
- Save annotated image
- Export field list to CSV
- Copy JSON to clipboard

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+O | Open Image |
| Ctrl+J | Open JSON |
| Ctrl+S | Save Overlay |
| Ctrl++ | Zoom In |
| Ctrl+- | Zoom Out |
| Ctrl+0 | Reset Zoom |
| Ctrl+F | Filter by Type |
| Escape | Clear Selection |

## Menu Bar

### File
- Open Image
- Open JSON
- Save Overlay As
- Recent Files
- Exit

### View
- Show/Hide Boxes
- Show/Hide Labels
- Show/Hide Confidence
- Fit to Window
- Zoom In/Out

### Tools
- Filter by Type
- Export Field List
- Copy JSON
- Compare Models

### Help
- Keyboard Shortcuts
- About

## Advanced Usage

### Batch View Multiple Forms
```python
from gui_viewer import DocAssistGUI
from PyQt5.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)

# Open multiple forms
windows = []
for form_path in form_list:
    window = DocAssistGUI(form_path + ".png", form_path + ".json")
    window.show()
    windows.append(window)

app.exec_()
```

### Custom Color Scheme
```python
FIELD_COLORS["custom_field"] = QColor(255, 0, 0, 180)
```

### Export with Custom Formatting
```python
from gui_viewer import export_fields_to_csv

export_fields_to_csv(fields, "fields.csv", include_metadata=True)
```

## Troubleshooting

### GUI Not Launching
```bash
# Check PyQt installation
python -c "from PyQt5.QtWidgets import QApplication; print('OK')"

# Reinstall if needed
pip uninstall PyQt5 PyQt6
pip install PyQt5
```

### Image Not Loading
- Supported formats: PNG, JPG, JPEG, PDF
- Check file path is correct
- Verify file permissions

### Bounding Boxes Misaligned
- Ensure image dimensions match JSON coordinates
- Check if image was resized after detection
- Use original image resolution

## API Reference

```python
from gui_viewer import DocAssistGUI, BoundingBoxScene

# Create viewer
viewer = DocAssistGUI("form.png", "fields.json")

# Programmatic control
viewer.load_files("new_form.png", "new_fields.json")
viewer.scene.set_fields(updated_fields)
viewer.update_view()
```

## Integration with Ensemble

```bash
# Run ensemble detection
python src/ensemble_client.py form.png --output ensemble.json

# View results
python src/gui_viewer.py --image form.png --json ensemble.json
```

The GUI will display:
- All validator fields (ground truth)
- Agreement status with trainer
- Confidence scores from both models
