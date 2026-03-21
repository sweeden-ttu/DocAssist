import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_field_extraction_output_format():
    sample_output = Path(__file__).parent.parent / "examples" / "sample_output.json"

    with open(sample_output) as f:
        result = json.load(f)

    assert "form_type" in result
    assert "fields" in result
    assert isinstance(result["fields"], list)

    for field in result["fields"]:
        assert "id" in field
        assert "type" in field
        assert "label" in field
        assert "bbox_2d" in field
        assert len(field["bbox_2d"]) == 4


def test_bbox_format():
    sample_output = Path(__file__).parent.parent / "examples" / "sample_output.json"

    with open(sample_output) as f:
        result = json.load(f)

    for field in result["fields"]:
        bbox = field["bbox_2d"]
        assert all(isinstance(x, (int, float)) for x in bbox)
        assert bbox[0] < bbox[2]
        assert bbox[1] < bbox[3]


def test_field_types():
    sample_output = Path(__file__).parent.parent / "examples" / "sample_output.json"

    with open(sample_output) as f:
        result = json.load(f)

    valid_types = [
        "text_input",
        "checkbox",
        "radio_button",
        "signature",
        "date",
        "currency",
        "ssn",
        "phone",
        "address",
        "name",
    ]

    for field in result["fields"]:
        assert field["type"] in valid_types


def test_confidence_scores():
    sample_output = Path(__file__).parent.parent / "examples" / "sample_output.json"

    with open(sample_output) as f:
        result = json.load(f)

    for field in result["fields"]:
        assert "confidence" in field
        assert 0.0 <= field["confidence"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
