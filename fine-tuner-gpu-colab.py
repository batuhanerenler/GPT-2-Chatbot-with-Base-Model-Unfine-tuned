import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
import torch
from google.colab import drive
from google.colab import files
import shutil
import matplotlib.pyplot as plt

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

# Calculate the number of trainable parameters before training
num_params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters before training: {num_params_before}")

# Set the pad_token to the eos_token
tokenizer.pad_token = tokenizer.eos_token

# Load the dataset
datasets = load_dataset("conv_ai_2")
train_data, val_data = train_test_split(datasets["train"].to_pandas(), test_size=0.1)

train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)

def tokenize_function(examples):
    dialog_texts = [' '.join([turn["text"] for turn in dialog]) for dialog in examples['dialog']]
    return tokenizer(dialog_texts, padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Create a data collator
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="chatbot_model",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=200,
    fp16=True,  # Enable mixed precision training
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Calculate the number of trainable parameters after training
num_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters after training: {num_params_after}")

# Save the trained model to Google Drive
drive.mount('/content/drive')
model_save_path = "/content/drive/MyDrive/chatbot_model"
trainer.save_model(model_save_path)

# Zip the saved model folder
shutil.make_archive("/content/drive/MyDrive/chatbot_model", 'zip', model_save_path)

# Download the zipped folder to your computer
files.download("/content/drive/MyDrive/chatbot_model.zip")

# Plot the training loss
plt.plot(trainer.state.log_history)
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
