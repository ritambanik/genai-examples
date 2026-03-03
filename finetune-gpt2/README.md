# Fine-tune GPT-2

This package provides examples and utilities for fine-tuning GPT-2 models on custom datasets.

## Features

- Data preprocessing and tokenization
- Fine-tuning scripts with configurable hyperparameters
- Model evaluation and inference utilities
- Support for different GPT-2 model sizes (small, medium, large, xl)

## Installation

```bash
# From the finetune-gpt2 directory
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

## Quick Start

### Basic Fine-tuning

```python
from finetune_gpt2 import GPT2Trainer

# Initialize trainer
trainer = GPT2Trainer(
    model_name="gpt2",
    output_dir="./output",
)

# Fine-tune on your dataset
trainer.train(
    train_data="path/to/train.txt",
    eval_data="path/to/eval.txt",
    num_epochs=3,
)
```

### Using the Command Line

```bash
# Train a model
python -m finetune_gpt2.train \
    --model_name gpt2 \
    --train_file data/train.txt \
    --output_dir ./output \
    --num_epochs 3

# Generate text with fine-tuned model
python -m finetune_gpt2.generate \
    --model_path ./output/checkpoint-final \
    --prompt "Once upon a time"
```

## Project Structure

```
finetune-gpt2/
├── finetune_gpt2/        # Main package
│   ├── __init__.py
│   ├── data.py           # Data loading and preprocessing
│   ├── model.py          # Model initialization
│   ├── trainer.py        # Training logic
│   ├── generate.py       # Text generation
│   └── utils.py          # Utility functions
├── tests/                # Unit tests
├── examples/             # Example scripts and notebooks
├── README.md
└── pyproject.toml
```

## Examples

See the [examples/](examples/) directory for:
- Jupyter notebooks with step-by-step tutorials
- Sample datasets and preprocessing scripts
- Advanced fine-tuning configurations

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- 8GB+ GPU memory (recommended for training)

## License

[Your chosen license]
