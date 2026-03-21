# DocAssist GitHub Project

## Overview
DocAssist is an IRS Form Field Extraction System that uses vision-language models (VLM) to detect and extract form fields with bounding box coordinates.

## Features
- [ ] Vision model-based form field detection
- [ ] Ensemble voting between CPU and MLX models
- [ ] Bounding box extraction with JSON output
- [ ] GUI overlay viewer for bounding boxes
- [ ] Episodic few-shot training pipeline
- [ ] Docling MCP integration for CLI tools

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       DocAssist System                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────┐         ┌─────────────────────┐       │
│  │  Linux (CPU-2-16)   │◄──────►│   Mac (MLX-4-16)    │       │
│  │  LM Studio :1234    │  HTTP  │  LM Studio :1234    │       │
│  │  Qwen2.5-VL-GGUF    │        │  Qwen2.5-VL-MLX    │       │
│  │  (Validator/GT)     │        │  (Trainer)          │       │
│  └─────────┬───────────┘         └─────────┬───────────┘       │
│            │                               │                    │
│            │        Ensemble Vote           │                    │
│            └───────────┬───────────────────┘                    │
│                        ▼                                        │
│              ┌─────────────────────┐                            │
│              │  Field Detection     │                            │
│              │  + Bounding Boxes   │                            │
│              │  + JSON Output      │                            │
│              └─────────────────────┘                            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Dual-Model Ensemble System

### Configuration
| Platform | Model | Endpoint | Purpose |
|----------|-------|----------|---------|
| Linux (CPU) | Qwen2.5-VL-7B-GGUF | localhost:1234 | Ground Truth Validator |
| Mac (MLX) | Qwen2.5-VL-7B-MLX | 192.168.0.13:1234 | Training/Testing |

### Voting Strategy
- Validator model (Linux/CPU) is the ground truth
- Both models must agree on field detection
- Trainer model outputs validated against validator

## Integration Components

### 1. LM Studio Models
- [ ] Qwen2.5-VL-7B-Instruct-GGUF (Linux)
- [ ] Qwen2.5-VL-7B-Instruct-MLX (Mac)

### 2. Docling Integration
- [ ] Docling MCP server setup
- [ ] CLI tool integration
- [ ] Python API wrapper

### 3. GUI Overlay Viewer
- [ ] PyQt/Tkinter form viewer
- [ ] JSON to overlay renderer
- [ ] Field type color coding
- [ ] Interactive field selection

### 4. Training Pipeline
- [ ] Episodic training framework
- [ ] LoRA fine-tuning configuration
- [ ] Training data generator

## Project Structure
```
DocAssist/
├── src/
│   ├── ensemble_client.py     # Dual-model voting
│   ├── form_detector.py      # Field detection
│   ├── field_extractor.py    # Field classification
│   ├── json_converter.py     # Format conversion
│   ├── episodic_trainer.py    # Few-shot training
│   ├── lmstudio_client.py    # LM Studio API
│   ├── docling_mcp.py        # Docling MCP integration
│   ├── gui_viewer.py         # Bounding box GUI
│   └── utils.py              # Utilities
├── configs/
│   ├── ensemble.yaml         # Dual-model config
│   ├── default.yaml           # Default settings
│   ├── qwen2.5vl.yaml        # Qwen config
│   └── episodic.yaml          # Training config
├── docs/
│   ├── API.md
│   ├── TRAINING.md
│   ├── IRS_FORMS.md
│   ├── DOCLING.md            # Docling integration
│   └── GUI.md                # GUI usage
├── tests/
├── examples/
├── README.md
└── pyproject.toml
```

## Milestones

### Milestone 1: Core Detection
- [ ] Set up dual-model ensemble
- [ ] Basic form field detection
- [ ] JSON output generation

### Milestone 2: Validation System
- [ ] Ensemble voting implementation
- [ ] Cross-validation pipeline
- [ ] Accuracy metrics

### Milestone 3: Docling Integration
- [ ] Docling MCP server
- [ ] CLI tool commands
- [ ] API integration

### Milestone 4: GUI Viewer
- [ ] Form overlay viewer
- [ ] Field type highlighting
- [ ] Interactive editing

### Milestone 5: Training Pipeline
- [ ] Episodic training setup
- [ ] LoRA fine-tuning
- [ ] Model evaluation

## Getting Started

```bash
# Clone repository
git clone https://github.com/sweeden-ttu/DocAssist.git
cd DocAssist

# Install dependencies
pip install -r requirements.txt

# Configure ensemble
cp configs/ensemble.example.yaml configs/ensemble.yaml

# Run form detection
python src/form_detector.py --input form.pdf --output output.json

# View with GUI
python src/gui_viewer.py --image form.png --json output.json
```

## Documentation
- [API Reference](docs/API.md)
- [Training Guide](docs/TRAINING.md)
- [IRS Forms](docs/IRS_FORMS.md)
- [Docling Integration](docs/DOCLING.md)
- [GUI Usage](docs/GUI.md)

## License
MIT License
