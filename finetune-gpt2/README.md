# Fine-tune GPT-2

Utilities and examples for fine-tuning GPT-2 using Hugging Face `transformers` and `datasets`.

## What this package does

- Creates an output workspace (`data/`, `results/`, `logs/`)
- Downloads and preprocesses a source PDF into train/validation text files
- Builds a Hugging Face `DatasetDict` with `train` and `validation` splits
- Tokenizes dataset splits and fine-tunes GPT-2 with `Trainer`
- Saves the final model and tokenizer under `results/final_model`

## Installation

```bash
# From this directory (finetune-gpt2)
pip install -e .

# Optional: development tools
pip install -e ".[dev]"
```

## Quick start

Run the example script:

```bash
python examples/basic_finetune.py
```

The script currently uses:

- `model_name="gpt2"`
- `output_dir="finetune-gpt2/output"`

You can also call the API directly:

```python
from finetune_gpt2.model import finetune_model

model, tokenizer = finetune_model(
    output_dir="finetune-gpt2/output",
    model_name="gpt2",
)
```

## Training workflow

`finetune_model(...)` in `finetune_gpt2/model.py` performs:

1. Create directory structure: `output/data`, `output/results`, `output/logs`
2. Build dataset via `prepare_dataset(...)` from `finetune_gpt2/data.py`
3. Tokenize each split from the `DatasetDict` (`train`, `validation`)
4. Train GPT-2 with `Trainer`
5. Save artifacts to `output/results/final_model`

## Output structure

After a run, your output directory looks like:

```text
output/
├── data/
│   ├── document.pdf
│   ├── train.txt
│   └── validation.txt
├── logs/
└── results/
    └── final_model/
```

## Project structure

```text
finetune-gpt2/
├── examples/
│   └── basic_finetune.py
├── finetune_gpt2/
│   ├── __init__.py
│   ├── data.py
│   └── model.py
├── tests/
├── README.md
└── pyproject.toml
```

## Notes

- GPU is used automatically when available.
- `prepare_dataset(...)` currently downloads a default PDF source and generates text files automatically.
- If dataset splits do not include a `text` column, tokenization raises a validation error.
