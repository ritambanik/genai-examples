"""
Model initialization and loading utilities.
"""

from pathlib import Path
from typing import Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import DatasetDict
from transformers import DataCollatorForLanguageModeling
import torch
from transformers import Trainer, TrainingArguments
from finetune_gpt2.data import prepare_dataset

def load_tokenizer_and_tokenize_dataset(
    model_name: str = "gpt2",
    dataset: Optional[DatasetDict] = None,
    max_length: int = 512,
):
    """
    Load the tokenizer and tokenize the dataset.

    Args:
        model_name: Name of the GPT-2 model to load (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
        dataset: Optional dataset to tokenize (if None, only the tokenizer is loaded)
        max_length: Maximum sequence length for tokenization
    Returns:
        Tokenizer and tokenized dataset (if dataset is provided)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        texts = [str(value) for value in examples["text"]]
        return tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)
    
    if dataset is not None:
        for split_name in dataset.keys():
            if "text" not in dataset[split_name].column_names:
                raise ValueError(f"Missing 'text' column in dataset split '{split_name}'.")
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenizer, tokenized_dataset
    return tokenizer, None


def create_datacollator(tokenizer):
    """
    Create a data collator for GPT-2 training.

    Args:
        tokenizer: HuggingFace tokenizer instance
    Returns:
        Data collator for GPT-2 training
    """
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )  # GPT-2 is a causal language model, so mlm=False


def finetune_model(
    output_dir: Union[str, Path],
    model_name: str = "gpt2",
):
    """
    Load a GPT-2 model from HuggingFace.

    Args:
        model_name: Name of the GPT-2 model to load (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
        load_in_8bit: Whether to load the model in 8-bit precision

    Returns:
        Loaded model and tokenizer
    """
    
    output_dir = Path(output_dir)
    
    dataset = prepare_dataset(data_path=output_dir / "data")
    
    tokenizer, tokens = load_tokenizer_and_tokenize_dataset(model_name, dataset=dataset)
    if tokens is None:
        raise ValueError("Tokenization failed because dataset is None.")
    data_collator = create_datacollator(tokenizer)
    
    train_args = TrainingArguments(
        output_dir=f"{output_dir}/results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir=f"{output_dir}/logs",)

    
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda" if torch.cuda.is_available() else "cpu")
    
        
    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        train_dataset=tokens["train"],
        eval_dataset=tokens["validation"],
        processing_class=tokenizer,
    )
    
    
    trainer.train()
    trainer.save_model(f"{output_dir}/results/final_model")
    tokenizer.save_pretrained(f"{output_dir}/results/final_model")
    return model, tokenizer


__all__ = ["finetune_model"]
