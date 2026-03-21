# DocAssist - IRS Form Field Extraction System

## Project Overview
DocAssist is a document understanding system that uses vision-language models to:
1. Detect and extract form fields from IRS forms
2. Generate structured JSON with bounding box coordinates
3. Support episodic few-shot training for custom form types

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   IRS Form PDF  │────▶│  Form Detector   │────▶│  Field Extractor│
│   (Input)       │     │  (FFDNet/Qwen)   │     │  (Qwen2.5-VL)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │   JSON Output   │
                                                 │  (Fields + BBox)│
                                                 └─────────────────┘
```

## Components

### 1. Form Field Detection
- **Model**: jbarrow/FFDNet-L or Qwen2.5-VL-7B
- **Task**: Detect form fields (text inputs, checkboxes, signatures)
- **Output**: Bounding boxes with confidence scores

### 2. Field Extraction
- **Model**: Qwen/Qwen2.5-VL-7B-Instruct
- **Task**: Classify fields and extract metadata
- **Output**: Structured JSON with field types, labels, coordinates

### 3. Episodic Training Pipeline
- **Framework**: SetFit + TRL
- **Task**: Few-shot learning for new form types
- **Output**: Fine-tuned models for specific IRS forms

## Supported IRS Forms
- [ ] Form 1040 (U.S. Individual Income Tax Return)
- [ ] Schedule C (Profit or Loss From Business)
- [ ] Form W-4 (Employee's Withholding Certificate)
- [ ] Form 1099 series (Various Income Forms)
- [ ] Form 8863 (Education Credits)

## Installation

```bash
cd /home/sweeden/projects/DocAssist
pip install -r requirements.txt
```

## Usage

```bash
# Extract fields from a single form
python src/extract_form.py --input forms/1040.pdf --output output/

# Batch processing
python src/batch_process.py --input-dir forms/ --output-dir output/

# Fine-tune for custom form
python src/train_episodic.py --config configs/irs1040.yaml
```

## Project Structure

```
DocAssist/
├── src/
│   ├── __init__.py
│   ├── form_detector.py      # Field detection using VLMs
│   ├── field_extractor.py    # Field classification & metadata
│   ├── json_converter.py     # Convert detections to JSON
│   ├── lmstudio_client.py   # LM Studio API wrapper
│   ├── episodic_trainer.py   # Few-shot training pipeline
│   └── utils.py              # Utility functions
├── configs/
│   ├── default.yaml          # Default configuration
│   ├── qwen2.5vl.yaml       # Qwen model config
│   └── episodic.yaml         # Training config
├── docs/
│   ├── README.md
│   ├── API.md
│   ├── TRAINING.md
│   └── IRS_FORMS.md
├── examples/
│   └── sample_output.json
├── tests/
│   └── test_extraction.py
└── models/
    └── (downloaded models)
```

## LM Studio Setup

1. Start LM Studio server on port 1234
2. Download models:
   - `Qwen/Qwen2.5-VL-7B-Instruct`
   - `jbarrow/FFDNet-L` (if available)
3. Configure endpoint in `configs/default.yaml`

## API Reference

See [API.md](docs/API.md) for detailed API documentation.

## Training

See [TRAINING.md](docs/TRAINING.md) for episodic training guide.

## License
MIT License
