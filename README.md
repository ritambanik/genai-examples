# GenAI Examples

Monorepo for practical generative AI examples, with each use case implemented as a standalone Python package.

## Repository layout

```text
genai-examples/
├── finetune-gpt2/         # GPT-2 fine-tuning example package
├── pyproject.toml         # Root project metadata
└── README.md
```

## Current package

### finetune-gpt2

The `finetune-gpt2` package demonstrates a complete GPT-2 fine-tuning flow:

- Create output directories (`data/`, `results/`, `logs/`)
- Prepare train/validation data files
- Build and tokenize a Hugging Face `DatasetDict`
- Train with `transformers.Trainer`
- Save model artifacts to `results/final_model`

See the package documentation: [finetune-gpt2/README.md](finetune-gpt2/README.md)

## Setup

```bash
# From repository root
pip install -e ".[dev]"

# Install example package
cd finetune-gpt2
pip install -e .
```

## Run the example

```bash
cd finetune-gpt2
python examples/basic_finetune.py
```

## Development commands

```bash
pytest
black .
flake8 .
```
