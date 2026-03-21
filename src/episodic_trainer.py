import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import random


@dataclass
class Episode:
    support_images: List[str] = field(default_factory=list)
    support_labels: List[str] = field(default_factory=list)
    query_images: List[str] = field(default_factory=list)
    query_labels: List[str] = field(default_factory=list)
    field_types: List[str] = field(default_factory=list)


@dataclass
class TrainingExample:
    image_path: str
    field_type: str
    label: str
    bbox: List[float]


class EpisodicTrainer:
    def __init__(self, n_way: int = 5, k_shot: int = 5, n_episodes: int = 100):
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_episodes = n_episodes
        self.field_types = [
            "text_input",
            "checkbox",
            "radio_button",
            "signature",
            "date",
            "currency",
            "ssn",
            "phone",
            "address",
        ]

    def create_episode(self, training_data: List[TrainingExample]) -> Episode:
        available_types = [
            t for t in self.field_types if any(e.field_type == t for e in training_data)
        ]

        if len(available_types) < self.n_way:
            selected_types = available_types
        else:
            selected_types = random.sample(available_types, self.n_way)

        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        field_types = []

        for field_type in selected_types:
            type_examples = [e for e in training_data if e.field_type == field_type]

            if len(type_examples) >= 2:
                support_example = random.choice(type_examples)
                query_example = random.choice(
                    [e for e in type_examples if e != support_example]
                )

                support_images.append(support_example.image_path)
                support_labels.append(support_example.label)
                query_images.append(query_example.image_path)
                query_labels.append(query_example.label)
                field_types.append(field_type)

        return Episode(
            support_images=support_images,
            support_labels=support_labels,
            query_images=query_images,
            query_labels=query_labels,
            field_types=field_types,
        )

    def generate_episodes(self, training_data: List[TrainingExample]) -> List[Episode]:
        episodes = []

        for _ in range(self.n_episodes):
            episode = self.create_episode(training_data)
            if episode.support_images:
                episodes.append(episode)

        return episodes

    def prepare_training_data(self, json_path: str) -> List[TrainingExample]:
        with open(json_path) as f:
            data = json.load(f)

        training_examples = []

        detections = data if isinstance(data, list) else [data]

        for detection in detections:
            for field_data in detection.get("fields", []):
                example = TrainingExample(
                    image_path=detection.get("image_path", ""),
                    field_type=field_data.get("type", "text_input"),
                    label=field_data.get("label", ""),
                    bbox=field_data.get("bbox_2d", []),
                )
                training_examples.append(example)

        return training_examples

    def export_episodes(self, episodes: List[Episode], output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        episodes_data = []

        for i, episode in enumerate(episodes):
            episode_dict = {
                "episode_id": i,
                "support": [
                    {"image": img, "label": label, "type": ft}
                    for img, label, ft in zip(
                        episode.support_images,
                        episode.support_labels,
                        episode.field_types,
                    )
                ],
                "query": [
                    {"image": img, "label": label, "type": ft}
                    for img, label, ft in zip(
                        episode.query_images, episode.query_labels, episode.field_types
                    )
                ],
            }
            episodes_data.append(episode_dict)

        with open(output_path / "episodes.json", "w") as f:
            json.dump(episodes_data, f, indent=2)

        with open(output_path / "training_metadata.json", "w") as f:
            metadata = {
                "n_way": self.n_way,
                "k_shot": self.k_shot,
                "n_episodes": len(episodes),
                "field_types": self.field_types,
            }
            json.dump(metadata, f, indent=2)

        print(f"Generated {len(episodes)} episodes")
        print(f"Output: {output_path}")

    def create_lora_config(self, output_path: str = "configs/lora_config.json"):
        lora_config = {
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "IMAGE_TEXT_TO_TEXT",
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(lora_config, f, indent=2)

        print(f"LoRA config saved to {output_path}")
        return lora_config


def main():
    parser = argparse.ArgumentParser(
        description="Episodic training for form field detection"
    )
    parser.add_argument("--input", "-i", required=True, help="Training data JSON")
    parser.add_argument(
        "--output", "-o", default="output/episodes", help="Output directory"
    )
    parser.add_argument(
        "--n-way", "-n", type=int, default=5, help="Number of classes per episode"
    )
    parser.add_argument(
        "--k-shot", "-k", type=int, default=5, help="Support examples per class"
    )
    parser.add_argument(
        "--episodes", "-e", type=int, default=100, help="Number of episodes"
    )
    parser.add_argument("--lora-config", "-l", help="Generate LoRA config file")

    args = parser.parse_args()

    trainer = EpisodicTrainer(
        n_way=args.n_way, k_shot=args.k_shot, n_episodes=args.episodes
    )

    training_data = trainer.prepare_training_data(args.input)
    print(f"Loaded {len(training_data)} training examples")

    episodes = trainer.generate_episodes(training_data)
    trainer.export_episodes(episodes, args.output)

    if args.lora_config:
        trainer.create_lora_config(args.lora_config)


if __name__ == "__main__":
    main()
