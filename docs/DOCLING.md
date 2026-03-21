# Docling Integration Guide

## Overview

Docling is an IBM-developed document understanding library that provides:
- Fast PDF parsing with layout analysis
- Table extraction
- Form field detection
- OCR support
- Multiple export formats (JSON, Markdown, Text)

## Conda Environment Setup

### Create Conda Environment

```bash
# Create conda environment
conda create -n docassist python=3.10 -y

# Activate environment
conda activate docassist

# Install PDFium binaries (required for Docling)
conda install -c anaconda pdfium-binaries -y

# Install docling
pip install docling docling-ibm-models

# Install PyQt for GUI
conda install -c conda-forge pyqt -y
```

### Alternative: Mamba for Faster Installation

```bash
# Install mamba (faster conda)
conda install mamba -n base -c conda-forge -y

# Create environment with mamba
mamba create -n docassist python=3.10 -c anaconda -c conda-forge -y
mamba activate docassist
mamba install pdfium-binaries pyqt -y
pip install docling docling-ibm-models
```

## CLI Usage

### Parse PDF to JSON
```bash
python docling_cli.py parse document.pdf --format json --output result.json
```

### Extract Tables
```bash
python docling_cli.py extract-tables document.pdf --output tables.json
```

### Extract Form Fields
```bash
python docling_cli.py extract-forms document.pdf --output forms.json
```

### Full Pipeline (Docling + VLM)
```bash
# First use Docling for structure
python docling_cli.py parse form.pdf --format json --output docling_output.json

# Then use VLM for field detection
python docling_cli.py detect form.png --output vlm_output.json

# Combine results
python docling_cli.py combine docling_output.json vlm_output.json --output combined.json
```

## Python API

```python
from src.docling_mcp import DoclingMCPServer

# Initialize
server = DoclingMCPServer()

# Parse PDF
result = server.parse_pdf("form.pdf", "json")

# Extract tables
tables = server.extract_tables("form.pdf")

# Extract forms
forms = server.extract_forms("form.pdf")
```

## MCP Server Mode

For integrating with Claude/other AI assistants:

```bash
python docling_cli.py mcp-server --port 8765
```

Then configure your AI assistant to use:
```
mcp-server: localhost:8765
```

## Features

### Layout Analysis
- Page segmentation
- Reading order detection
- Header/footer identification

### Table Extraction
- Structured table detection
- Cell extraction
- Row/column alignment

### Form Processing
- Form field detection
- Field type classification
- Fillable field identification

### Export Formats
- JSON (full structure)
- Markdown (readable)
- Text (plain text)
- HTML (formatted)

## Advanced Usage

### Custom Pipeline Options
```python
from docling.datamodel.pipeline_options import PdfPipelineOptions

options = PdfPipelineOptions(
    do_table_structure=True,
    do_formula_detection=True,
    do_ocr=True,
    do_picture_classification=True
)

converter = DocumentConverter(pipeline_options=options)
```

### Batch Processing
```python
from pathlib import Path

for pdf_file in Path("forms/").glob("*.pdf"):
    result = converter.convert(pdf_file)
    output = pdf_file.stem + ".json"
    with open(output, 'w') as f:
        json.dump(result.document.export_to_dict(), f)
```

## Combining Docling + VLM

The recommended workflow combines Docling's structure understanding with VLM's semantic understanding:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Docling   │────▶│  Structure  │────▶│   VLM       │
│   (PDF)     │     │   (JSON)    │     │   (Fields)  │
└─────────────┘     └─────────────┘     └─────────────┘
```

1. **Docling** extracts layout structure, tables, and form positions
2. **VLM** validates field types and adds semantic labels
3. **Combined** output provides both structural and semantic information

## Troubleshooting

### Import Errors
```bash
pip install --upgrade docling
```

### Slow Processing
- Use `--workers 4` for parallel processing
- Enable GPU acceleration if available

### Memory Issues
- Process large PDFs in batches
- Use `--max-pages 10` to limit processing
