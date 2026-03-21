#!/usr/bin/env python3
"""
DocAssist - IRS Form Field Extraction
Usage: python run.py [command] [options]
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from form_detector import FormDetector, save_results, create_field_summary
from field_extractor import FieldExtractor
from json_converter import JSONConverter, FieldVisualizer
from episodic_trainer import EpisodicTrainer
from utils import setup_directories, load_config


def cmd_extract(args):
    detector = FormDetector(args.lmstudio_url)
    input_path = Path(args.input)

    if input_path.suffix.lower() == ".pdf":
        results = detector.process_pdf(
            str(input_path), str(input_path.parent / "images"), args.form_type
        )
    else:
        results = [detector.detect_fields(str(input_path), args.form_type)]

    save_results(results, args.output)
    create_field_summary(results, args.summary)
    print(f"Extraction complete! Output: {args.output}")


def cmd_convert(args):
    converter = JSONConverter()
    converter.convert_file(args.input, args.output, args.format)
    print(f"Conversion complete! Output: {args.output}")


def cmd_visualize(args):
    import json

    with open(args.input) as f:
        detection = json.load(f)

    if isinstance(detection, list):
        detection = detection[0] if detection else {}

    visualizer = FieldVisualizer()
    visualizer.draw_bounding_boxes(args.image, detection, args.output)
    print(f"Visualization saved: {args.output}")


def cmd_train(args):
    trainer = EpisodicTrainer(
        n_way=args.n_way, k_shot=args.k_shot, n_episodes=args.episodes
    )

    training_data = trainer.prepare_training_data(args.input)
    print(f"Loaded {len(training_data)} training examples")

    episodes = trainer.generate_episodes(training_data)
    trainer.export_episodes(episodes, args.output)

    if args.lora_config:
        trainer.create_lora_config(args.lora_config)

    print("Training episode generation complete!")


def cmd_enhance(args):
    extractor = FieldExtractor()

    with open(args.input) as f:
        detection = json.load(f)

    if isinstance(detection, list):
        detection = detection[0] if detection else {}

    enhanced = extractor.extract_with_context(detection, args.form_type)

    with open(args.output, "w") as f:
        import json

        json.dump(enhanced, f, indent=2)

    if args.template:
        template = extractor.generate_fill_template(enhanced)
        with open(args.template, "w") as f:
            json.dump(template, f, indent=2)
        print(f"Fill template: {args.template}")

    print(f"Enhanced output: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="DocAssist - IRS Form Field Extraction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    extract_parser = subparsers.add_parser("extract", help="Extract fields from form")
    extract_parser.add_argument(
        "--input", "-i", required=True, help="Input PDF or image"
    )
    extract_parser.add_argument("--output", "-o", default="output/form_fields.json")
    extract_parser.add_argument("--summary", "-s", default="output/field_summary.json")
    extract_parser.add_argument("--form-type", default="IRS Form", help="Form type")
    extract_parser.add_argument("--lmstudio-url", default="http://localhost:1234")

    convert_parser = subparsers.add_parser("convert", help="Convert output format")
    convert_parser.add_argument("--input", "-i", required=True, help="Input JSON")
    convert_parser.add_argument("--output", "-o", required=True, help="Output JSON")
    convert_parser.add_argument(
        "--format", "-f", choices=["standard", "coco", "yolo"], default="standard"
    )

    viz_parser = subparsers.add_parser("visualize", help="Visualize bounding boxes")
    viz_parser.add_argument("--input", "-i", required=True, help="Detection JSON")
    viz_parser.add_argument("--image", required=True, help="Source image")
    viz_parser.add_argument("--output", "-o", required=True, help="Output image")

    train_parser = subparsers.add_parser("train", help="Generate training episodes")
    train_parser.add_argument("--input", "-i", required=True, help="Training data JSON")
    train_parser.add_argument("--output", "-o", default="output/episodes")
    train_parser.add_argument("--n-way", "-n", type=int, default=5)
    train_parser.add_argument("--k-shot", "-k", type=int, default=5)
    train_parser.add_argument("--episodes", "-e", type=int, default=100)
    train_parser.add_argument("--lora-config", "-l", help="Generate LoRA config")

    enhance_parser = subparsers.add_parser(
        "enhance", help="Enhance detection with context"
    )
    enhance_parser.add_argument("--input", "-i", required=True)
    enhance_parser.add_argument("--output", "-o", default="output/enhanced.json")
    enhance_parser.add_argument("--template", "-t", help="Generate fill template")
    enhance_parser.add_argument("--form-type", default="IRS Form 1040")

    args = parser.parse_args()

    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "convert":
        cmd_convert(args)
    elif args.command == "visualize":
        cmd_visualize(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "enhance":
        cmd_enhance(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
