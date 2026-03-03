"""
Basic example of fine-tuning GPT-2 on a custom dataset.

This is a template script showing the general structure.
Uncomment and implement the sections as needed.
"""

# from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
# from finetune_gpt2.data import prepare_dataset

from finetune_gpt2.model import finetune_model


def main():
    """Main training function."""
    print("GPT-2 Fine-tuning Example")
    print("=" * 50)

    # Configuration
    model_name = "gpt2"
    output_dir = "finetune-gpt2/output"

    print(f"Model: {model_name}")
    print(f"Output directory: {output_dir}")

    # Fine-tune the model
    finetune_model(output_dir=output_dir, model_name=model_name)

if __name__ == "__main__":
    main()
