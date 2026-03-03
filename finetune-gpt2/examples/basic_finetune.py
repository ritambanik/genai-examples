"""
Basic example of fine-tuning GPT-2 on a custom dataset.

This is a template script showing the general structure.
Uncomment and implement the sections as needed.
"""

# from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
# from finetune_gpt2.data import prepare_dataset


def main():
    """Main training function."""
    print("GPT-2 Fine-tuning Example")
    print("=" * 50)

    # Configuration
    model_name = "gpt2"
    train_file = "data/train.txt"
    output_dir = "./output"

    print(f"Model: {model_name}")
    print(f"Training data: {train_file}")
    print(f"Output directory: {output_dir}")

    # TODO: Implement training logic
    # 1. Load tokenizer and model
    # 2. Prepare dataset
    # 3. Configure training arguments
    # 4. Initialize trainer
    # 5. Train model
    # 6. Save final model

    print("\nTraining not yet implemented. See finetune_gpt2 package modules.")


if __name__ == "__main__":
    main()
