#!/usr/bin/env python3
"""
Docling CLI Tool for DocAssist

Usage:
    python docling_cli.py parse <pdf_file> [--format json|markdown|text]
    python docling_cli.py tables <pdf_file>
    python docling_cli.py forms <pdf_file>
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
except ImportError as e:
    DOCLING_AVAILABLE = False
    print(f"Error: Docling not installed: {e}")
    print("Run: conda activate taxenv && pip install docling docling-ibm-models")


class DoclingCLI:
    def __init__(self):
        if DOCLING_AVAILABLE:
            self.converter = DocumentConverter(
                allowed_formats=[
                    InputFormat.PDF,
                    InputFormat.IMAGE,
                    InputFormat.DOCX,
                    InputFormat.HTML,
                    InputFormat.PPTX,
                ]
            )
        else:
            self.converter = None

    def parse(self, pdf_path: str, output_format: str = "json", output_file: Optional[str] = None):
        """Parse PDF and export to specified format"""
        if not self.converter:
            print("Error: Docling not available")
            return 1

        print(f"Parsing: {pdf_path}")
        result = self.converter.convert(pdf_path)

        if output_format == "json":
            output = result.document.export_to_dict()
        elif output_format == "text":
            output = {"text": result.document.export_to_text()}
        elif output_format == "markdown":
            output = {"markdown": result.document.export_to_markdown()}
        else:
            output = result.document.export_to_dict()

        if output_file:
            with open(output_file, "w") as f:
                json.dump(output, f, indent=2)
            print(f"Output saved: {output_file}")
        else:
            print(json.dumps(output, indent=2))

        return 0

    def tables(self, pdf_path: str, output_file: Optional[str] = None):
        """Extract tables from PDF"""
        if not self.converter:
            print("Error: Docling not available")
            return 1

        print(f"Extracting tables from: {pdf_path}")
        result = self.converter.convert(pdf_path)

        tables = []
        for table in result.document.tables:
            tables.append(
                {
                    "row_count": len(table.rows) if hasattr(table, "rows") else 0,
                    "col_count": len(table.cols) if hasattr(table, "cols") else 0,
                    "table": table.export_to_dict()
                    if hasattr(table, "export_to_dict")
                    else str(table),
                }
            )

        output = {"tables": tables, "count": len(tables)}

        if output_file:
            with open(output_file, "w") as f:
                json.dump(output, f, indent=2)
            print(f"Extracted {len(tables)} tables to: {output_file}")
        else:
            print(json.dumps(output, indent=2))

        return 0

    def forms(self, pdf_path: str, output_file: Optional[str] = None):
        """Extract form fields from PDF"""
        if not self.converter:
            print("Error: Docling not available")
            return 1

        print(f"Extracting forms from: {pdf_path}")
        result = self.converter.convert(pdf_path)

        doc_dict = result.document.export_to_dict()

        form_data = {
            "document_name": doc_dict.get("name", ""),
            "pages": len(doc_dict.get("pages", [])),
            "texts": len(doc_dict.get("texts", [])),
            "tables": len(doc_dict.get("tables", [])),
            "form_items": doc_dict.get("form_items", []),
            "key_value_items": doc_dict.get("key_value_items", []),
        }

        if output_file:
            with open(output_file, "w") as f:
                json.dump(form_data, f, indent=2)
            print(f"Form data saved: {output_file}")
        else:
            print(json.dumps(form_data, indent=2))

        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Docling CLI for DocAssist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Parse command
    parse_parser = subparsers.add_parser("parse", help="Parse PDF document")
    parse_parser.add_argument("pdf_file", help="PDF file to parse")
    parse_parser.add_argument(
        "--format", "-f", choices=["json", "markdown", "text"], default="json", help="Output format"
    )
    parse_parser.add_argument("--output", "-o", help="Output file")

    # Tables command
    tables_parser = subparsers.add_parser("tables", help="Extract tables")
    tables_parser.add_argument("pdf_file", help="PDF file")
    tables_parser.add_argument("--output", "-o", help="Output file")

    # Forms command
    forms_parser = subparsers.add_parser("forms", help="Extract form fields")
    forms_parser.add_argument("pdf_file", help="PDF file")
    forms_parser.add_argument("--output", "-o", help="Output file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    cli = DoclingCLI()

    if args.command == "parse":
        return cli.parse(args.pdf_file, args.format, args.output)
    elif args.command == "tables":
        return cli.tables(args.pdf_file, args.output)
    elif args.command == "forms":
        return cli.forms(args.pdf_file, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
