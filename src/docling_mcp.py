# Docling MCP Integration Guide

## Overview

Docling is an IBM-developed document understanding library that provides fast, accurate PDF parsing and layout analysis. Integrating Docling as an MCP (Model Context Protocol) server enables CLI tools and programmatic access to Docling's capabilities.

## Installation

```bash
pip install docling
pip install docling-ibm-models
```

## MCP Server Setup

### Option 1: Standalone MCP Server

```python
# src/docling_mcp.py
"""
Docling MCP Server for DocAssist
Provides document parsing as a tool callable via MCP protocol
"""

import json
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdf_backend import PyPdfDocumentBackend
import sys

class DoclingMCPServer:
    def __init__(self):
        self.converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.IMAGE,
                InputFormat.DOCX,
                InputFormat.HTML,
                InputFormat.PPTX
            ]
        )
    
    def parse_pdf(self, pdf_path: str, output_format: str = "json") -> Dict[str, Any]:
        """Parse PDF and extract content with layout"""
        result = self.converter.convert(pdf_path)
        
        if output_format == "json":
            return result.document.export_to_dict()
        elif output_format == "text":
            return {"text": result.document.export_to_text()}
        elif output_format == "markdown":
            return {"markdown": result.document.export_to_markdown()}
        else:
            return result.document.export_to_dict()
    
    def extract_tables(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables from PDF"""
        result = self.converter.convert(pdf_path)
        tables = []
        
        for element in result.document.elements:
            if hasattr(element, 'table'):
                tables.append({
                    "table": element.table,
                    "bbox": element.bbox if hasattr(element, 'bbox') else None
                })
        
        return tables
    
    def extract_forms(self, pdf_path: str) -> Dict[str, Any]:
        """Extract form fields from PDF"""
        result = self.converter.convert(pdf_path)
        
        forms = {
            "fields": [],
            "annotations": []
        }
        
        for element in result.document.iterables():
            if hasattr(element, 'type'):
                if element.type == "form":
                    forms["fields"].append({
                        "type": "form_field",
                        "label": getattr(element, 'label', ''),
                        "value": getattr(element, 'value', ''),
                        "bbox": element.bbox if hasattr(element, 'bbox') else None
                    })
        
        return forms


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Docling MCP Server")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--mode", choices=["server", "cli"], default="cli")
    
    args = parser.parse_args()
    
    server = DoclingMCPServer()
    
    if args.mode == "server":
        print(f"Starting Docling MCP server on {args.host}:{args.port}")
        # Start MCP server (requires mcp library)
        # mcp.server.run("localhost", 8765, server)
    else:
        print("Docling MCP CLI mode")
        print("Available commands:")
        print("  parse - Parse PDF to JSON")
        print("  tables - Extract tables")
        print("  forms - Extract form fields")


if __name__ == "__main__":
    main()
