from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Load the dataset from a JSON Lines file
dataset = load_dataset('json', data_files='fine_tuning_dataset.jsonl')

# Specify the model name and load its tokenizer
model_name = 'distilgpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the padding token to the end-of-sequence token if it is not defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Define a function to tokenize input examples
def tokenize_function(examples):
    tokenized_inputs = tokenizer(
        examples['text'],  # Tokenize the 'text' field
        examples['answer'],  # Tokenize the 'answer' field
        padding='max_length',  # Pad to the maximum length
        truncation=True,  # Truncate inputs that exceed the max length
        max_length=512,  # Set maximum sequence length to 512
        return_attention_mask=True,  # Return attention masks
    )
    # Create labels for the model by copying input IDs
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

# Apply the tokenization function to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Load the pre-trained language model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',  # Directory to save training results
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=1,  # Training batch size per device
    per_device_eval_batch_size=1,  # Evaluation batch size per device
    gradient_accumulation_steps=8,  # Accumulate gradients over 8 steps
    learning_rate=5e-5,  # Learning rate
    weight_decay=0.01,  # Weight decay for regularization
    logging_steps=10,  # Log training info every 10 steps
    save_steps=50,  # Save the model every 50 steps
    evaluation_strategy="no",  # Disable evaluation during training
    save_total_limit=2,  # Limit the number of saved checkpoints
    load_best_model_at_end=False,  # Do not load the best model at the end
)

# Initialize the Trainer class
trainer = Trainer(
    model=model,  # The model to train
    args=training_args,  # Training arguments
    train_dataset=tokenized_dataset['train'],  # Training dataset
    tokenizer=tokenizer,  # Tokenizer for the model
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
trainer.save_model('./fine-tuned-model')
tokenizer.save_pretrained('./fine-tuned-model')