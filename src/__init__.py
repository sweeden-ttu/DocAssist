# DocAssist - IRS Form Field Extraction System
__version__ = "1.0.0"

from .lmstudio_client import LMStudioClient
from .form_detector import FormDetector
from .field_extractor import FieldExtractor
from .json_converter import JSONConverter, FieldVisualizer
from .episodic_trainer import EpisodicTrainer, Episode, TrainingExample

__all__ = [
    "LMStudioClient",
    "FormDetector",
    "FieldExtractor",
    "JSONConverter",
    "FieldVisualizer",
    "EpisodicTrainer",
    "Episode",
    "TrainingExample",
]
