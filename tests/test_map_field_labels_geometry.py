"""Unit tests for map_field_labels_geometry (PDF rects + label heuristics)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from map_field_labels_geometry import (
    PdfRect,
    field_coords_to_rect,
    iou,
    pick_label_for_field,
)


def test_field_coords_to_rect():
    r = field_coords_to_rect(10, 20, 30, 5)
    assert r.left == 10 and r.bottom == 20 and r.right == 40 and r.top == 25


def test_pick_label_left_strip():
    field = field_coords_to_rect(200, 50, 100, 20)
    words = [
        (PdfRect(50, 52, 120, 68), "First"),
        (PdfRect(125, 52, 190, 68), "Last"),
    ]
    label, score, method, _ = pick_label_for_field(field, words)
    assert "First" in label and "Last" in label
    assert method == "left_strip"
    assert score > 0


def test_pick_label_above_strip():
    field = field_coords_to_rect(200, 50, 80, 20)
    words = [
        (PdfRect(200, 72, 280, 88), "Above"),
    ]
    label, score, method, _ = pick_label_for_field(field, words)
    assert "Above" in label
    assert method == "above_strip"


def test_iou_overlap():
    a = PdfRect(0, 0, 10, 10)
    b = PdfRect(5, 5, 15, 15)
    assert iou(a, b) > 0
