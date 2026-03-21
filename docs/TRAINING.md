# DocAssist Training Guide

## Overview

This guide covers training DocAssist models for form field detection using episodic few-shot learning.

## Setup

```bash
pip install setfit peft transformers accelerate bitsandbytes
```

## Training Pipeline

### 1. Data Preparation

Prepare your labeled form field data in JSON format:

```json
{
  "form_type": "IRS Form 1040",
  "page": 1,
  "image_path": "path/to/image.png",
  "fields": [
    {
      "type": "text_input",
      "label": "First name",
      "bbox_2d": [x1, y1, x2, y2]
    }
  ]
}
```

### 2. Generate Training Episodes

```python
from src.episodic_trainer import EpisodicTrainer

trainer = EpisodicTrainer(n_way=5, k_shot=5, n_episodes=100)
training_data = trainer.prepare_training_data("labeled_data.json")
episodes = trainer.generate_episodes(training_data)
trainer.export_episodes(episodes, "output/episodes")
```

### 3. Fine-tune with LoRA

```python
from peft import LoraConfig, get_peft_model
from transformers import Qwen2_5_VLForConditionalGeneration

# Load base model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="IMAGE_TEXT_TO_TEXT"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

### 4. Training Loop

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="output/model",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    logging_steps=10,
    save_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=episodes_dataset,
)

trainer.train()
```

## Episodic Learning Details

### What is Episodic Training?

Episodic training mimics how humans learn new tasks - by learning from a small number of examples (episodes), where each episode contains:
- **Support set**: Examples to learn from
- **Query set**: Examples to evaluate on

### Episode Structure

```
Episode {
  support_images: List[str]      # Training examples
  support_labels: List[str]      # Labels for support
  query_images: List[str]       # Test examples
  query_labels: List[str]       # Ground truth for query
  field_types: List[str]        # Field types in this episode
}
```

### N-way K-shot Learning

- **N-way**: Number of different classes per episode
- **K-shot**: Number of examples per class in support set

Example: 5-way 5-shot means:
- 5 different field types per episode
- 5 examples of each type for training

## Model Variants

### Qwen2.5-VL (Recommended)
- Best for general form understanding
- Excellent bounding box accuracy
- Supports OCR + structure understanding

### FFDNet
- Specialized for form field detection
- Faster inference
- Lower memory requirements

## Validation

### Metrics

```python
def calculate_metrics(predictions, ground_truth):
    metrics = {
        "precision": calculate_precision(predictions, ground_truth),
        "recall": calculate_recall(predictions, ground_truth),
        "f1": calculate_f1(predictions, ground_truth),
        "iou": calculate_mean_iou(predictions, ground_truth)
    }
    return metrics
```

### Evaluation Dataset

Keep a separate validation set not used in episodic training:
- 80% training / 20% validation split
- Evaluate every 50 episodes
- Track metrics over time

## Troubleshooting

### Out of Memory
- Reduce batch size
- Use 4-bit quantization
- Reduce image resolution

### Poor Accuracy
- Increase number of episodes
- Add more training examples
- Try different N-way K-shot values

### Slow Training
- Use gradient checkpointing
- Enable mixed precision
- Use flash attention

## Next Steps

1. Start with 5-way 5-shot training
2. Evaluate on validation set
3. Increase difficulty (fewer shots, more classes)
4. Fine-tune on specific IRS form types
