#!/usr/bin/env python3
"""
DocAssist Web Dashboard

Flask-based web interface for IRS form field extraction using VLMs.
"""

import os
import json
import base64
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

from flask import Flask, request, jsonify, render_template, send_file, abort
import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"

app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    template_folder=str(TEMPLATES_DIR),
)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = tempfile.mkdtemp(prefix="docassist_")


@dataclass
class HealthStatus:
    validator_available: bool
    trainer_available: bool
    validator_url: str
    trainer_url: str
    validator_model: Optional[str] = None
    trainer_model: Optional[str] = None


@dataclass
class PipelineStep:
    key: str
    title: str
    description: str
    completed: bool
    details: str = ""


class LMStudioAPI:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.models_url = f"{self.base_url}/v1/models"
        self.chat_url = f"{self.base_url}/v1/chat/completions"

    def get_status(self) -> Dict[str, Any]:
        try:
            response = httpx.get(self.models_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [m["id"] for m in data.get("data", [])]
                return {"available": True, "models": models}
            return {"available": False, "models": []}
        except Exception as e:
            return {"available": False, "models": [], "error": str(e)}

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def extract_fields(self, image_path: str) -> Dict[str, Any]:
        image_base64 = self.encode_image(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                    {
                        "type": "text",
                        "text": """You are a document analysis expert for IRS tax forms.
Analyze this form and identify all fillable fields.
For each field provide:
- type: text_input, checkbox, signature, date, currency, select
- label: descriptive name
- bbox_2d: [x1, y1, x2, y2] coordinates
- page: page number
Return JSON with fields array.""",
                    },
                ],
            }
        ]

        payload = {
            "model": "local-model",
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 4096,
        }

        try:
            import time

            start = time.time()
            response = httpx.post(self.chat_url, json=payload, timeout=180)
            response.raise_for_status()
            elapsed = time.time() - start

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            json_str = content.strip()
            for marker in ["```json", "```"]:
                if json_str.startswith(marker):
                    json_str = json_str[len(marker) :]
                if json_str.endswith(marker):
                    json_str = json_str[: -len(marker)]

            parsed = json.loads(json_str.strip())
            return {"success": True, "data": parsed, "inference_time": elapsed, "raw": content}
        except Exception as e:
            return {"success": False, "error": str(e)}


validator_api = LMStudioAPI("http://192.168.0.15:1234")
trainer_api = LMStudioAPI("http://192.168.0.13:1234")

IRS_OCR_ANALYSIS_PATH = PROJECT_ROOT / "annotations" / "irs" / "f1040_ocr_analysis.json"
IRS_OVERLAYS_DIR = PROJECT_ROOT / "annotations" / "irs" / "overlays"
IRS_ANNOTATED_DIR = PROJECT_ROOT / "annotations" / "irs" / "annotated"


def _build_pipeline_steps() -> List[Dict[str, Any]]:
    """Build the 3-step processing view shown in the dashboard."""
    ocr_exists = IRS_OCR_ANALYSIS_PATH.exists() and IRS_OCR_ANALYSIS_PATH.stat().st_size > 0
    overlay_count = len(list(IRS_OVERLAYS_DIR.glob("page_*_multi_overlay.png"))) if IRS_OVERLAYS_DIR.exists() else 0
    annotated_count = len(list(IRS_ANNOTATED_DIR.glob("page_*_multi_annotated.png"))) if IRS_ANNOTATED_DIR.exists() else 0

    steps = [
        PipelineStep(
            key="step1_detect",
            title="Step 1 - Field Detection",
            description="Run VLM extraction to produce OCR analysis JSON.",
            completed=ocr_exists,
            details=str(IRS_OCR_ANALYSIS_PATH) if ocr_exists else "OCR analysis file not found yet.",
        ),
        PipelineStep(
            key="step2_post_process",
            title="Step 2 - Post-Processing",
            description="Apply post-processing to refine labels/answers before overlay generation.",
            completed=ocr_exists,
            details="Uses post_process_ocr_with_vlm.py output as source for overlays.",
        ),
        PipelineStep(
            key="step3_overlay",
            title="Step 3 - Transparent Overlay",
            description="Generate transparent overlays and annotated page previews.",
            completed=overlay_count > 0,
            details=f"{overlay_count} overlay(s), {annotated_count} annotated page(s)",
        ),
    ]
    return [asdict(step) for step in steps]


def _find_annotation_dir_for_form(form_name: str) -> Optional[Path]:
    """
    Best-effort match from extracted form name (e.g. f1040_docling) to an
    annotations subdirectory that has page PNGs and overlay PNGs.
    """
    annotations_root = PROJECT_ROOT / "annotations"
    if not annotations_root.exists():
        return None

    form_key = form_name.lower().replace("_docling", "")
    for child in annotations_root.iterdir():
        if not child.is_dir():
            continue
        page_pngs = sorted(child.glob("page_*.png"))
        overlay_pngs = sorted((child / "overlays").glob("page_*_multi_overlay.png"))
        if not page_pngs or not overlay_pngs:
            continue
        if any(form_key in p.name.lower() for p in child.glob("*.json")):
            return child

    # Fallback: if only one annotations set has overlays, use it.
    candidates = []
    for child in annotations_root.iterdir():
        if child.is_dir() and list((child / "overlays").glob("page_*_multi_overlay.png")):
            candidates.append(child)
    if len(candidates) == 1:
        return candidates[0]
    return None


def _get_form_overlay_pages(form_name: str) -> List[Dict[str, Any]]:
    """Return page asset URLs for transparent overlay preview by form."""
    annotations_dir = _find_annotation_dir_for_form(form_name)
    if not annotations_dir:
        return []

    overlays_dir = annotations_dir / "overlays"
    annotated_dir = annotations_dir / "annotated"
    pages: List[Dict[str, Any]] = []
    for overlay in sorted(overlays_dir.glob("page_*_multi_overlay.png")):
        parts = overlay.stem.split("_")
        if len(parts) < 4:
            continue
        try:
            page_num = int(parts[1])
        except ValueError:
            continue

        page_png = annotations_dir / f"page_{page_num}.png"
        annotated_png = annotated_dir / f"page_{page_num}_multi_annotated.png"
        if not page_png.exists():
            continue

        pages.append(
            {
                "page": page_num,
                "page_image_url": f"/api/form-assets/{annotations_dir.name}/page_{page_num}.png",
                "overlay_image_url": f"/api/form-assets/{annotations_dir.name}/overlays/{overlay.name}",
                "annotated_image_url": (
                    f"/api/form-assets/{annotations_dir.name}/annotated/{annotated_png.name}"
                    if annotated_png.exists()
                    else None
                ),
            }
        )
    return pages


@app.route("/")
def index():
    return render_template("index.html", pipeline_steps=_build_pipeline_steps())


@app.route("/api/pipeline/steps")
def pipeline_steps():
    return jsonify({"steps": _build_pipeline_steps()})


@app.route("/api/health")
def health():
    v_status = validator_api.get_status()
    t_status = trainer_api.get_status()

    return jsonify(
        {
            "validator": {
                "available": v_status["available"],
                "url": validator_api.base_url,
                "models": v_status.get("models", []),
            },
            "trainer": {
                "available": t_status["available"],
                "url": trainer_api.base_url,
                "models": t_status.get("models", []),
            },
        }
    )


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    image_base64 = validator_api.encode_image(filepath)

    return jsonify(
        {
            "success": True,
            "filename": file.filename,
            "size": os.path.getsize(filepath),
            "image_preview": f"data:image/jpeg;base64,{image_base64[:1000]}...",
        }
    )


@app.route("/api/detect", methods=["POST"])
def detect():
    data = request.get_json()
    if not data or "image_path" not in data:
        return jsonify({"error": "No image path provided"}), 400

    image_path = data["image_path"]
    if not os.path.exists(image_path):
        return jsonify({"error": "File not found"}), 404

    use_validator = data.get("use_validator", True)
    use_trainer = data.get("use_trainer", False)

    results = {"validator": None, "trainer": None}

    if use_validator:
        results["validator"] = validator_api.extract_fields(image_path)

    if use_trainer:
        results["trainer"] = trainer_api.extract_fields(image_path)

    return jsonify(results)


@app.route("/api/extract-form", methods=["POST"])
def extract_form():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    use_ensemble = request.form.get("ensemble", "false").lower() == "true"

    results = {"validator": None, "trainer": None}

    v_result = validator_api.extract_fields(filepath)
    if v_result["success"]:
        results["validator"] = v_result

    if use_ensemble:
        t_result = trainer_api.extract_fields(filepath)
        if t_result["success"]:
            results["trainer"] = t_result

    os.remove(filepath)

    return jsonify(
        {
            "success": True,
            "filename": file.filename,
            "results": results,
            "ensemble_enabled": use_ensemble,
        }
    )


@app.route("/api/forms")
def list_forms():
    forms_dir = Path("/home/sweeden/projects/docling_data/tax_packet/docling_extracted")
    if not forms_dir.exists():
        return jsonify({"forms": [], "count": 0})

    forms = []
    for f in forms_dir.glob("*_docling.json"):
        forms.append(
            {
                "name": f.stem,
                "path": str(f),
                "size": f.stat().st_size,
            }
        )

    return jsonify({"forms": forms, "count": len(forms)})


@app.route("/api/forms/<path:form_name>/overlay-pages")
def form_overlay_pages(form_name: str):
    pages = _get_form_overlay_pages(form_name)
    return jsonify({"form": form_name, "pages": pages, "count": len(pages)})


@app.route("/api/form-assets/<path:asset_path>")
def form_asset(asset_path: str):
    """
    Serve annotation assets for previews:
      /api/form-assets/<subdir>/page_1.png
      /api/form-assets/<subdir>/overlays/page_1_multi_overlay.png
    """
    base = (PROJECT_ROOT / "annotations").resolve()
    target = (base / asset_path).resolve()
    if not str(target).startswith(str(base)) or not target.exists() or not target.is_file():
        abort(404)
    return send_file(str(target))


def create_app():
    return app


if __name__ == "__main__":
    print("=" * 60)
    print("DocAssist Web Dashboard")
    print("=" * 60)
    print(f"Validator: http://192.168.0.15:1234")
    print(f"Trainer:   http://192.168.0.13:1234")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)
