import os
import json
import base64
from pathlib import Path
from typing import Optional
import httpx
from PIL import Image
import io


class LMStudioClient:
    def __init__(self, base_url: str = "http://localhost:1234"):
        self.base_url = base_url.rstrip("/")
        self.chat_endpoint = f"{self.base_url}/v1/chat/completions"
        self.models_endpoint = f"{self.base_url}/v1/models"

    def is_available(self) -> bool:
        try:
            response = httpx.get(self.models_endpoint, timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def chat(
        self,
        messages: list,
        model: str = "lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        try:
            response = httpx.post(self.chat_endpoint, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except httpx.HTTPError as e:
            raise RuntimeError(f"LM Studio API error: {e}")

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def extract_form_fields(
        self, image_path: str, prompt: Optional[str] = None
    ) -> dict:
        if prompt is None:
            prompt = """You are a document analysis expert specializing in IRS tax forms.
Analyze this form image and identify all fillable/form fields.
For each field, provide:
1. The field type (text_input, checkbox, signature, date, currency)
2. A descriptive label for the field
3. The bounding box coordinates as [x1, y1, x2, y2] in pixels

Return your response as a valid JSON object with this exact structure:
{
  "form_type": "IRS Form 1040",
  "page": 1,
  "fields": [
    {
      "id": "field_001",
      "type": "text_input",
      "label": "First name and middle initial",
      "bbox_2d": [x1, y1, x2, y2],
      "confidence": 0.95,
      "fillable": true
    }
  ]
}

Only output valid JSON, no markdown formatting."""

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

        response = self.chat(messages)
        return self._parse_json_response(response)

    def _parse_json_response(self, response: str) -> dict:
        json_str = response.strip()

        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]

        json_str = json_str.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON response: {e}\nResponse: {response[:500]}"
            )
