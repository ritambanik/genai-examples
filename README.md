# GenAI Examples

A collection of practical generative AI use case examples, organized as separate packages.

## Project Structure

This project uses a monorepo structure with individual packages for each use case:

```
genai-examples/
├── finetune-gpt2/         # GPT-2 fine-tuning examples
├── ...                    # Additional use cases
├── pyproject.toml         # Root project configuration
└── README.md
```

## Available Packages

### finetune-gpt2
Fine-tuning examples for GPT-2 models. See [finetune-gpt2/README.md](finetune-gpt2/README.md) for details.

## Installation

### Development Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd genai-examples
```

2. Install the root package with development dependencies:
```bash
pip install -e ".[dev]"
```

3. Install individual packages:
```bash
cd finetune-gpt2
pip install -e .
```

## Adding New Packages

To add a new use case package:

1. Create a new directory under the root (e.g., `new-use-case/`)
2. Set up the package structure with `pyproject.toml`, `README.md`, and source code
3. Update this README to list the new package

## Development

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
black .
```

### Linting
```bash
flake8 .
```

## License

[Your chosen license]
