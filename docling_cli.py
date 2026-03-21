#!/usr/bin/env python3
"""
DocAssist CLI with Docling Integration

Usage:
    python docling_cli.py parse <pdf_file> [--format json|markdown|text]
    python docling_cli.py tables <pdf_file>
    python docling_cli.py forms <pdf_file>
    python docling_cli.py detect <image_file> --model qwen
    python docling_cli.py ensemble <image_file>
    python docling_cli.py gui --image <file> --json <file>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat

    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    print("Warning: Docling not installed. Run: pip install docling")

try:
    sys.path.insert(0, str(Path(__file__).parent))
    from lmstudio_client import LMStudioClient
    from ensemble_client import EnsembleValidator

    LMSTUDIO_AVAILABLE = True
except ImportError:
    LMSTUDIO_AVAILABLE = False


class DocAssistCLI:
    def __init__(self):
        self.docling_server = None
        self.lmstudio_client = None
        self.ensemble = None

        if DOCLING_AVAILABLE:
            self.docling_server = self._init_docling()

    def _init_docling(self):
        if not DOCLING_AVAILABLE:
            return None
        from src.docling_mcp import DoclingMCPServer

        return DoclingMCPServer()

    def _init_lmstudio(self, endpoint: str = "http://localhost:1234"):
        if not LMSTUDIO_AVAILABLE:
            return None
        return LMStudioClient(endpoint)

    def cmd_parse(self, pdf_file: str, format: str = "json", output: Optional[str] = None):
        """Parse PDF using Docling"""
        if not self.docling_server:
            print("Error: Docling not available")
            return 1

        print(f"Parsing {pdf_file}...")
        result = self.docling_server.parse_pdf(pdf_file, format)

        if output:
            with open(output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Output saved to {output}")
        else:
            print(json.dumps(result, indent=2))

        return 0

    def cmd_tables(self, pdf_file: str, output: Optional[str] = None):
        """Extract tables from PDF"""
        if not self.docling_server:
            print("Error: Docling not available")
            return 1

        print(f"Extracting tables from {pdf_file}...")
        tables = self.docling_server.extract_tables(pdf_file)

        result = {"tables": tables, "count": len(tables)}

        if output:
            with open(output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Extracted {len(tables)} tables to {output}")
        else:
            print(json.dumps(result, indent=2))

        return 0

    def cmd_forms(self, pdf_file: str, output: Optional[str] = None):
        """Extract form fields from PDF"""
        if not self.docling_server:
            print("Error: Docling not available")
            return 1

        print(f"Extracting forms from {pdf_file}...")
        forms = self.docling_server.extract_forms(pdf_file)

        if output:
            with open(output, "w") as f:
                json.dump(forms, f, indent=2)
            print(f"Form fields saved to {output}")
        else:
            print(json.dumps(forms, indent=2))

        return 0

    def cmd_detect(self, image_file: str, model: str = "qwen", output: Optional[str] = None):
        """Detect form fields using VLM"""
        if not LMSTUDIO_AVAILABLE:
            print("Error: LM Studio client not available")
            return 1

        client = self._init_lmstudio()
        print(f"Detecting fields in {image_file} using {model}...")

        result = client.extract_form_fields(image_file)

        if output:
            with open(output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Detection results saved to {output}")
        else:
            print(json.dumps(result, indent=2))

        return 0

    def cmd_ensemble(self, image_file: str, output: Optional[str] = None):
        """Run ensemble detection with both models"""
        if not LMSTUDIO_AVAILABLE:
            print("Error: LM Studio client not available")
            return 1

        validator_url = "http://localhost:1234"
        trainer_url = "http://192.168.0.13:1234"

        print(f"Running ensemble detection...")
        print(f"  Validator: {validator_url}")
        print(f"  Trainer: {trainer_url}")

        ensemble = EnsembleValidator(validator_url, trainer_url)
        result = ensemble.detect_fields(image_file)

        if output:
            with open(output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Ensemble results saved to {output}")
        else:
            print(json.dumps(result, indent=2))

        return 0

    def cmd_gui(self, image_file: Optional[str] = None, json_file: Optional[str] = None):
        """Launch GUI viewer"""
        from gui_viewer import main as gui_main

        sys.argv = ["gui_viewer.py"]
        if image_file:
            sys.argv.extend(["--image", image_file])
        if json_file:
            sys.argv.extend(["--json", json_file])
        gui_main()
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="DocAssist CLI with Docling integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Parse PDF to JSON:
    %(prog)s parse document.pdf --format json --output result.json
  
  Extract tables:
    %(prog)s tables document.pdf --output tables.json
  
  Detect form fields:
    %(prog)s detect form.png --output fields.json
  
  Ensemble detection:
    %(prog)s ensemble form.png --output ensemble.json
  
  Launch GUI:
    %(prog)s gui --image form.png --json fields.json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Parse command
    parse_parser = subparsers.add_parser("parse", help="Parse PDF using Docling")
    parse_parser.add_argument("pdf_file", help="PDF file to parse")
    parse_parser.add_argument(
        "--format", "-f", choices=["json", "markdown", "text"], default="json"
    )
    parse_parser.add_argument("--output", "-o", help="Output file")

    # Tables command
    tables_parser = subparsers.add_parser("tables", help="Extract tables from PDF")
    tables_parser.add_argument("pdf_file", help="PDF file to extract tables from")
    tables_parser.add_argument("--output", "-o", help="Output file")

    # Forms command
    forms_parser = subparsers.add_parser("forms", help="Extract form fields from PDF")
    forms_parser.add_argument("pdf_file", help="PDF file to extract forms from")
    forms_parser.add_argument("--output", "-o", help="Output file")

    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Detect form fields using VLM")
    detect_parser.add_argument("image_file", help="Image file to analyze")
    detect_parser.add_argument("--model", "-m", default="qwen", help="Model to use")
    detect_parser.add_argument("--output", "-o", help="Output file")

    # Ensemble command
    ensemble_parser = subparsers.add_parser("ensemble", help="Run ensemble detection")
    ensemble_parser.add_argument("image_file", help="Image file to analyze")
    ensemble_parser.add_argument("--output", "-o", help="Output file")

    # GUI command
    gui_parser = subparsers.add_parser("gui", help="Launch GUI viewer")
    gui_parser.add_argument("--image", "-i", help="Image file to display")
    gui_parser.add_argument("--json", "-j", help="JSON detection file")

    args = parser.parse_args()

    cli = DocAssistCLI()

    if args.command == "parse":
        return cli.cmd_parse(args.pdf_file, args.format, args.output)
    elif args.command == "tables":
        return cli.cmd_tables(args.pdf_file, args.output)
    elif args.command == "forms":
        return cli.cmd_forms(args.pdf_file, args.output)
    elif args.command == "detect":
        return cli.cmd_detect(args.image_file, args.model, args.output)
    elif args.command == "ensemble":
        return cli.cmd_ensemble(args.image_file, args.output)
    elif args.command == "gui":
        return cli.cmd_gui(args.image, args.json)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
