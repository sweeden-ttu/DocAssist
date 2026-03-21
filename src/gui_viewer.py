#!/usr/bin/env python3
"""
DocAssist GUI - Bounding Box Overlay Viewer

A PyQt5-based GUI for viewing form fields with bounding box overlays.
"""

import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import argparse

try:
    from PyQt5.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QFileDialog,
        QScrollArea,
        QGraphicsView,
        QGraphicsScene,
        QSlider,
        QComboBox,
        QCheckBox,
        QGroupBox,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
        QSplitter,
    )
    from PyQt5.QtCore import Qt, QRectF, QPointF
    from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QTransform

    PYQT5_AVAILABLE = True
except ImportError:
    try:
        from PyQt6.QtWidgets import (
            QApplication,
            QMainWindow,
            QWidget,
            QVBoxLayout,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QFileDialog,
            QScrollArea,
            QGraphicsView,
            QGraphicsScene,
            QSlider,
            QComboBox,
            QCheckBox,
            QGroupBox,
            QTableWidget,
            QTableWidgetItem,
            QHeaderView,
            QSplitter,
        )
        from PyQt6.QtCore import Qt, QRectF, QPointF
        from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QTransform

        PYQT5_AVAILABLE = True
    except ImportError:
        PYQT5_AVAILABLE = False
        print("Warning: PyQt5/PyQt6 not installed. GUI will not be available.")
        print("Install with: pip install PyQt5")


FIELD_COLORS = {
    "text_input": QColor(0, 100, 255, 180),
    "checkbox": QColor(0, 200, 100, 180),
    "radio_button": QColor(100, 200, 0, 180),
    "signature": QColor(200, 0, 200, 180),
    "date": QColor(255, 165, 0, 180),
    "currency": QColor(255, 255, 0, 180),
    "ssn": QColor(0, 255, 255, 180),
    "phone": QColor(255, 0, 100, 180),
    "address": QColor(100, 0, 200, 180),
    "name": QColor(0, 150, 150, 180),
    "default": QColor(128, 128, 128, 180),
}


class BoundingBoxScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fields = []
        self.show_boxes = True
        self.show_labels = True
        self.show_confidence = True
        self.zoom_level = 1.0

    def set_fields(self, fields: List[Dict[str, Any]]):
        self.fields = fields

    def set_show_options(self, boxes=True, labels=True, confidence=True):
        self.show_boxes = boxes
        self.show_labels = labels
        self.show_confidence = confidence

    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)

        if not self.show_boxes:
            return

        for field in self.fields:
            bbox = field.get("bbox_2d", [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            field_type = field.get("type", "default")
            color = FIELD_COLORS.get(field_type, FIELD_COLORS["default"])

            pen = QPen(color, 2)
            painter.setPen(pen)
            painter.setBrush(color)

            rect = QRectF(x1, y1, x2 - x1, y2 - y1)
            painter.drawRect(rect)

            if self.show_labels:
                label = f"{field.get('type', 'field')}"
                if self.show_confidence:
                    conf = field.get("confidence", 0)
                    label += f" ({conf:.0%})"

                font = QFont("Arial", 10)
                painter.setFont(font)
                text_color = QColor(255, 255, 255)
                painter.setPen(text_color)

                label_rect = QRectF(x1, y1 - 20, 150, 20)
                painter.fillRect(label_rect, color)
                painter.drawText(label_rect, Qt.AlignLeft | Qt.AlignVCenter, label)


class FormOverlayViewer(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.zoom_factor = 1.25

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.scale(self.zoom_factor, self.zoom_factor)
        else:
            self.scale(1.0 / self.zoom_factor, 1.0 / self.zoom_factor)


class FieldTableModel:
    def __init__(self, fields: List[Dict[str, Any]]):
        self.fields = fields

    def rowCount(self):
        return len(self.fields)

    def columnCount(self):
        return 5

    def data(self, row, column):
        field = self.fields[row]
        columns = ["id", "type", "label", "bbox_2d", "confidence"]
        key = columns[column]
        value = field.get(key, "")

        if key == "bbox_2d" and isinstance(value, list):
            value = str(value)
        elif key == "confidence" and isinstance(value, float):
            value = f"{value:.2%}"

        return str(value)

    def headerData(self, column):
        headers = ["ID", "Type", "Label", "Bounding Box", "Confidence"]
        return headers[column]


class DocAssistGUI(QMainWindow):
    def __init__(self, image_path: Optional[str] = None, json_path: Optional[str] = None):
        super().__init__()
        self.image_path = image_path
        self.json_path = json_path
        self.fields = []
        self.pixmap_item = None

        self.init_ui()

        if image_path and json_path:
            self.load_files(image_path, json_path)

    def init_ui(self):
        self.setWindowTitle("DocAssist - Form Field Viewer")
        self.setGeometry(100, 100, 1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        left_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(left_splitter)

        self.scene = BoundingBoxScene()
        self.graphics_view = FormOverlayViewer(self.scene)

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.graphics_view)
        scroll_area.setWidgetResizable(True)
        left_splitter.addWidget(scroll_area)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        left_splitter.addWidget(right_panel)

        left_splitter.setSizes([1000, 400])

        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)

        button_layout = QHBoxLayout()
        self.btn_load_image = QPushButton("Load Image")
        self.btn_load_json = QPushButton("Load JSON")
        self.btn_save_overlay = QPushButton("Save Overlay")
        button_layout.addWidget(self.btn_load_image)
        button_layout.addWidget(self.btn_load_json)
        button_layout.addWidget(self.btn_save_overlay)
        controls_layout.addLayout(button_layout)

        options_layout = QHBoxLayout()
        self.chk_boxes = QCheckBox("Show Boxes")
        self.chk_labels = QCheckBox("Show Labels")
        self.chk_confidence = QCheckBox("Show Confidence")
        self.chk_boxes.setChecked(True)
        self.chk_labels.setChecked(True)
        self.chk_confidence.setChecked(True)
        options_layout.addWidget(self.chk_boxes)
        options_layout.addWidget(self.chk_labels)
        options_layout.addWidget(self.chk_confidence)
        controls_layout.addLayout(options_layout)

        right_layout.addWidget(controls_group)

        zoom_group = QGroupBox("Zoom")
        zoom_layout = QHBoxLayout(zoom_group)
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(10)
        self.zoom_slider.setMaximum(500)
        self.zoom_slider.setValue(100)
        self.lbl_zoom = QLabel("100%")
        zoom_layout.addWidget(self.zoom_slider)
        zoom_layout.addWidget(self.lbl_zoom)
        right_layout.addWidget(zoom_group)

        filter_group = QGroupBox("Filter by Type")
        filter_layout = QVBoxLayout(filter_group)
        self.type_combo = QComboBox()
        self.type_combo.addItem("All Types", None)
        for field_type in FIELD_COLORS.keys():
            if field_type != "default":
                self.type_combo.addItem(field_type.title().replace("_", " "), field_type)
        filter_layout.addWidget(self.type_combo)
        right_layout.addWidget(filter_group)

        table_group = QGroupBox("Field Details")
        table_layout = QVBoxLayout(table_group)
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["ID", "Type", "Label", "Bounding Box", "Confidence"])
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        table_layout.addWidget(self.table)
        right_layout.addWidget(table_group)

        self.statusBar().showMessage("Ready")

        self.btn_load_image.clicked.connect(self.on_load_image)
        self.btn_load_json.clicked.connect(self.on_load_json)
        self.btn_save_overlay.clicked.connect(self.on_save_overlay)
        self.chk_boxes.toggled.connect(self.update_view)
        self.chk_labels.toggled.connect(self.update_view)
        self.chk_confidence.toggled.connect(self.update_view)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        self.type_combo.currentIndexChanged.connect(self.on_filter_changed)
        self.table.cellClicked.connect(self.on_table_clicked)

    def load_files(self, image_path: str, json_path: str):
        self.image_path = image_path
        self.json_path = json_path

        with open(json_path) as f:
            data = json.load(f)

        if isinstance(data, list):
            self.fields = data[0].get("fields", []) if data else []
        else:
            self.fields = data.get("fields", [])

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.statusBar().showMessage(f"Failed to load image: {image_path}")
            return

        self.scene.clear()
        self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())

        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.scene.set_fields(self.fields)

        self.graphics_view.setScene(self.scene)
        self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        self.update_table()
        self.update_view()

        self.statusBar().showMessage(f"Loaded: {image_path} with {len(self.fields)} fields")

    def update_view(self):
        self.scene.set_show_options(
            self.chk_boxes.isChecked(), self.chk_labels.isChecked(), self.chk_confidence.isChecked()
        )
        self.scene.update()

    def update_table(self):
        self.table.setRowCount(len(self.fields))

        for row, field in enumerate(self.fields):
            self.table.setItem(row, 0, QTableWidgetItem(field.get("id", "")))
            self.table.setItem(row, 1, QTableWidgetItem(field.get("type", "")))
            self.table.setItem(row, 2, QTableWidgetItem(field.get("label", "")))
            self.table.setItem(row, 3, QTableWidgetItem(str(field.get("bbox_2d", []))))
            conf = field.get("confidence", 0)
            self.table.setItem(row, 4, QTableWidgetItem(f"{conf:.2%}"))

    def on_load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "", "Images (*.png *.jpg *.jpeg *.pdf)"
        )
        if path:
            if self.json_path:
                self.load_files(path, self.json_path)
            else:
                self.image_path = path
                self.statusBar().showMessage(f"Image loaded: {path}")

    def on_load_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load JSON", "", "JSON Files (*.json)")
        if path:
            if self.image_path:
                self.load_files(self.image_path, path)
            else:
                self.json_path = path
                self.statusBar().showMessage(f"JSON loaded: {path}")

    def on_save_overlay(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Overlay", "", "PNG Image (*.png)")
        if path:
            image = self.scene.toImage()
            image.save(path)
            self.statusBar().showMessage(f"Overlay saved: {path}")

    def on_zoom_changed(self, value):
        zoom = value / 100.0
        self.lbl_zoom.setText(f"{value}%")
        self.graphics_view.resetTransform()
        self.graphics_view.scale(zoom, zoom)

    def on_filter_changed(self, index):
        selected_type = self.type_combo.currentData()

        if selected_type is None:
            filtered_fields = self.fields
        else:
            filtered_fields = [f for f in self.fields if f.get("type") == selected_type]

        self.scene.set_fields(filtered_fields)
        self.scene.update()

    def on_table_clicked(self, row, column):
        if row < len(self.fields):
            field = self.fields[row]
            bbox = field.get("bbox_2d", [])
            if len(bbox) == 4:
                self.graphics_view.centerOn((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def main():
    parser = argparse.ArgumentParser(description="DocAssist Form Field Viewer")
    parser.add_argument("--image", "-i", help="Image file to display")
    parser.add_argument("--json", "-j", help="JSON detection file")

    args = parser.parse_args()

    if not PYQT5_AVAILABLE:
        print("Error: PyQt5 or PyQt6 required for GUI")
        print("Install with: pip install PyQt5")
        return 1

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = DocAssistGUI(args.image, args.json)
    window.show()

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
