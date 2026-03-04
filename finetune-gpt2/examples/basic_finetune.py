"""
Basic example of fine-tuning GPT-2 on a custom dataset.

This is a template script showing the general structure.
Uncomment and implement the sections as needed.
"""

# from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
# from finetune_gpt2.data import prepare_dataset

import sys

import torch

from finetune_gpt2.model import finetune_model


output_dir = "finetune-gpt2/output"

def finetune_model():
     # Configuration
    model_name = "gpt2"

    print(f"Model: {model_name}")
    print(f"Output directory: {output_dir}")

    # Fine-tune the model
    finetune_model(output_dir=output_dir, model_name=model_name)
    
    
def generate_response(prompt: str) -> str:
    # load the fine-tuned model and tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(f"{output_dir}/results/final_model")
    tokenizer = AutoTokenizer.from_pretrained(f"{output_dir}/results/final_model")
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    pad_token_id = tokenizer.eos_token_id
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, attention_mask=attention_mask, pad_token_id=pad_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def main():
    """Main training function."""
    print("GPT-2 Fine-tuning Example")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "finetune":
        finetune_model()
    elif len(sys.argv) > 1 and sys.argv[1] == "generate_response":
        print(generate_response(sys.argv[2]))
    else:
        print("No valid command provided. Use 'finetune' to start fine-tuning.")




if __name__ == "__main__":
    main()
