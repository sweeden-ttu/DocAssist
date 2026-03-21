#!/usr/bin/env python3
"""
DocAssist Ensemble Client

Dual-model ensemble system using both local (CPU) and remote (Mac MLX) models.
Validator model (Linux/CPU) is ground truth.
"""

import json
import base64
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import httpx
from dataclasses import dataclass


@dataclass
class ModelResult:
    """Result from a single model inference"""

    fields: List[Dict[str, Any]]
    model_name: str
    endpoint: str
    confidence: float
    inference_time: float
    raw_response: str


class LMStudioClient:
    """Client for LM Studio API"""

    def __init__(self, base_url: str = "http://localhost:1234", model: str = "local-model"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.chat_endpoint = f"{self.base_url}/v1/chat/completions"
        self.models_endpoint = f"{self.base_url}/v1/models"

    def is_available(self) -> bool:
        try:
            response = httpx.get(self.models_endpoint, timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def extract_form_fields(self, image_path: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        if prompt is None:
            prompt = """You are a document analysis expert specializing in IRS tax forms.
Analyze this form image and identify all fillable/form fields.
For each field, provide:
1. The field type (text_input, checkbox, signature, date, currency)
2. A descriptive label for the field
3. The bounding box coordinates as [x1, y1, x2, y2] in pixels
Return valid JSON with form_type, page, and fields array."""

        image_base64 = self.encode_image(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 4096,
            "stream": False,
        }

        try:
            import time

            start_time = time.time()

            response = httpx.post(self.chat_endpoint, json=payload, timeout=180)
            response.raise_for_status()
            result = response.json()

            inference_time = time.time() - start_time

            content = result["choices"][0]["message"]["content"]

            json_str = content.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.startswith("```"):
                json_str = json_str[3:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]

            parsed = json.loads(json_str.strip())

            return {"result": parsed, "inference_time": inference_time, "raw_response": content}

        except httpx.HTTPError as e:
            raise RuntimeError(f"LM Studio API error: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}")


class EnsembleValidator:
    """
    Ensemble validator using dual-model system.

    Architecture:
    - Validator (Linux/CPU): Ground truth model, always trusted
    - Trainer (Mac/MLX): Secondary model, validated against validator
    """

    def __init__(
        self,
        validator_url: str = "http://localhost:1234",
        trainer_url: str = "http://192.168.0.13:1234",
        validator_model: str = "Qwen/Qwen2.5-VL-7B-Instruct-GGUF",
        trainer_model: str = "lmstudio-community/Qwen2.5-VL-7B-Instruct-4bit-MLX",
    ):
        self.validator = LMStudioClient(validator_url, validator_model)
        self.trainer = LMStudioClient(trainer_url, trainer_model)

        self.validation_config = {
            "voting_strategy": "validator_ground_truth",
            "confidence_threshold": 0.85,
            "validator_weight": 1.0,
            "trainer_weight": 0.5,
            "require_agreement": False,
        }

    def is_validator_available(self) -> bool:
        return self.validator.is_available()

    def is_trainer_available(self) -> bool:
        return self.trainer.is_available()

    def detect_fields(self, image_path: str) -> Dict[str, Any]:
        """
        Run ensemble detection on an image.
        Returns validated results from validator model.
        """
        results = self._run_ensemble(image_path)
        return self._validate_results(results)

    def _run_ensemble(self, image_path: str) -> Dict[str, Any]:
        """Run both models and collect results"""
        results = {
            "validator": None,
            "trainer": None,
            "agreed_fields": [],
            "disagreed_fields": [],
            "trainer_only_fields": [],
        }

        if self.is_validator_available():
            try:
                validator_result = self.validator.extract_form_fields(image_path)
                results["validator"] = ModelResult(
                    fields=validator_result["result"].get("fields", []),
                    model_name=self.validator.model,
                    endpoint=self.validator.base_url,
                    confidence=sum(
                        f.get("confidence", 0) for f in validator_result["result"].get("fields", [])
                    )
                    / max(len(validator_result["result"].get("fields", [])), 1),
                    inference_time=validator_result["inference_time"],
                    raw_response=validator_result["raw_response"],
                )
            except Exception as e:
                print(f"Validator error: {e}")

        if self.is_trainer_available():
            try:
                trainer_result = self.trainer.extract_form_fields(image_path)
                results["trainer"] = ModelResult(
                    fields=trainer_result["result"].get("fields", []),
                    model_name=self.trainer.model,
                    endpoint=self.trainer.base_url,
                    confidence=sum(
                        f.get("confidence", 0) for f in trainer_result["result"].get("fields", [])
                    )
                    / max(len(trainer_result["result"].get("fields", [])), 1),
                    inference_time=trainer_result["inference_time"],
                    raw_response=trainer_result["raw_response"],
                )
            except Exception as e:
                print(f"Trainer error: {e}")

        return results

    def _validate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate results - validator is ground truth.
        Trainer results are compared against validator.
        """
        validator_fields = results["validator"].fields if results["validator"] else []
        trainer_fields = results["trainer"].fields if results["trainer"] else []

        validated_fields = []

        for v_field in validator_fields:
            field_id = v_field.get("id", "")
            v_bbox = v_field.get("bbox_2d", [])
            v_type = v_field.get("type", "")
            v_confidence = v_field.get("confidence", 0)

            matching_trainer_field = self._find_matching_field(v_field, trainer_fields)

            if matching_trainer_field:
                t_confidence = matching_trainer_field.get("confidence", 0)
                if t_confidence >= self.validation_config["confidence_threshold"]:
                    validated_fields.append(
                        {
                            **v_field,
                            "validated_by_trainer": True,
                            "trainer_confidence": t_confidence,
                            "agreement": True,
                        }
                    )
                else:
                    validated_fields.append(
                        {
                            **v_field,
                            "validated_by_trainer": True,
                            "trainer_confidence": t_confidence,
                            "agreement": False,
                            "note": "Trainer confidence below threshold",
                        }
                    )
            else:
                validated_fields.append(
                    {
                        **v_field,
                        "validated_by_trainer": False,
                        "agreement": None,
                        "note": "No matching trainer field",
                    }
                )

        extra_trainer_fields = [
            t for t in trainer_fields if not self._find_matching_field(t, validator_fields)
        ]

        ensemble_result = {
            "form_type": validator_fields[0].get("form_type", "Unknown")
            if validator_fields
            else "Unknown",
            "page": 1,
            "fields": validated_fields,
            "metadata": {
                "ensemble_mode": "validator_ground_truth",
                "validator_model": self.validator.model if results["validator"] else None,
                "trainer_model": self.trainer.model if results["trainer"] else None,
                "validator_available": results["validator"] is not None,
                "trainer_available": results["trainer"] is not None,
                "total_validator_fields": len(validator_fields),
                "total_trainer_fields": len(trainer_fields),
                "extra_trainer_fields": len(extra_trainer_fields),
                "validation_config": self.validation_config,
            },
            "trainer_extra_fields": extra_trainer_fields,
        }

        return ensemble_result

    def _find_matching_field(
        self, field: Dict[str, Any], candidate_fields: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find matching field based on IoU threshold"""
        field_bbox = field.get("bbox_2d", [])
        if len(field_bbox) != 4:
            return None

        best_match = None
        best_iou = 0.5

        for candidate in candidate_fields:
            candidate_bbox = candidate.get("bbox_2d", [])
            if len(candidate_bbox) != 4:
                continue

            iou = self._calculate_iou(field_bbox, candidate_bbox)

            if iou > best_iou:
                best_iou = iou
                best_match = candidate

        return best_match

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def compare_models(self, image_path: str) -> Dict[str, Any]:
        """
        Compare outputs from both models without voting.
        Useful for analyzing model agreement.
        """
        results = self._run_ensemble(image_path)

        comparison = {
            "validator_fields": results["validator"].fields if results["validator"] else [],
            "trainer_fields": results["trainer"].fields if results["trainer"] else [],
            "validator_confidence": results["validator"].confidence if results["validator"] else 0,
            "trainer_confidence": results["trainer"].confidence if results["trainer"] else 0,
            "validator_inference_time": results["validator"].inference_time
            if results["validator"]
            else 0,
            "trainer_inference_time": results["trainer"].inference_time
            if results["trainer"]
            else 0,
            "agreement_rate": 0,
            "disagreements": [],
        }

        if results["validator"] and results["trainer"]:
            agreements = 0
            total = len(results["validator"].fields)

            for v_field in results["validator"].fields:
                if self._find_matching_field(v_field, results["trainer"].fields):
                    agreements += 1

            comparison["agreement_rate"] = agreements / total if total > 0 else 0

        return comparison


def main():
    import argparse

    parser = argparse.ArgumentParser(description="DocAssist Ensemble Validator")
    parser.add_argument("image", help="Image file to analyze")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--validator-url", default="http://localhost:1234")
    parser.add_argument("--trainer-url", default="http://192.168.0.13:1234")
    parser.add_argument("--compare", action="store_true", help="Compare model outputs")

    args = parser.parse_args()

    ensemble = EnsembleValidator(validator_url=args.validator_url, trainer_url=args.trainer_url)

    print("DocAssist Ensemble Validator")
    print(f"Validator: {args.validator_url}")
    print(f"Trainer: {args.trainer_url}")
    print()

    if args.compare:
        print("Running model comparison...")
        result = ensemble.compare_models(args.image)
        print(f"Validator confidence: {result['validator_confidence']:.2%}")
        print(f"Trainer confidence: {result['trainer_confidence']:.2%}")
        print(f"Agreement rate: {result['agreement_rate']:.2%}")
    else:
        print("Running ensemble detection...")
        result = ensemble.detect_fields(args.image)
        print(f"Detected {len(result['fields'])} fields")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
