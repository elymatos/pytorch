import json
import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling


# Load and prepare dataset
def load_construction_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    # Create prompt and response style dataset
    examples = [{
        "prompt": f"Sentence: {item['sentence']}\nAnalyze the construction.",
        "response": f"Construction: {item['construction']}\nSemantic Class: {item['semantic_class']}\nInformation Packaging: {item['information_packaging']}"
    } for item in data]
    return Dataset.from_list(examples)


# Tokenization function for individual examples
def tokenize_function(example, tokenizer):
    full_text = f"{example['prompt']}\n{example['response']}"
    return tokenizer(full_text, truncation=True, padding="max_length", max_length=512)


def main():
    model_checkpoint = "tiiuae/falcon-rw-1b"  # Change this to any causal model
    json_path = "constructional_examples_llm.json"
    output_dir = "./construction_model"

    # Initialize tokenizer with padding token
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    # Update model embeddings for new tokens
    model.resize_token_embeddings(len(tokenizer))

    # Load and prepare dataset
    print("Loading dataset...")
    dataset = load_construction_data(json_path)
    print(f"Dataset loaded with {len(dataset)} examples")

    # Apply tokenization to each example individually
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=False  # Process one example at a time
    )

    # Remove original text columns
    tokenized_dataset = tokenized_dataset.remove_columns(['prompt', 'response'])

    # Verify dataset structure
    print(f"Dataset columns: {tokenized_dataset.column_names}")
    print(f"Sample tokenized example: {tokenized_dataset[0]}")

    # Prepare data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="no"
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Train and save model
    print("Starting training...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training completed and model saved!")


if __name__ == "__main__":
    main()