"""
Training an Large Language Model (LLM) in Danish clinical text data
"""
import os
import torch
from transformers import LlamaForCausalLM, AutoTokenizer, TrainingArguments
from tqdm import tqdm
from typing import Dict
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def llama_test(df_dk: DataFrame, llm_model: str, batch_size: int, num_epochs: int):
    """
    Train the Llama3 model in danish clinical text data
    """
    # Preprocess the text data
    tokenizer = AutoTokenizer.from_pretrained(llm_model, padding=True,
                                              padding_side="right", pad_token="[PAD]",
                                              add_eos_token=True, add_bos_token=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Input_ids and attention_mask
    def preprocess_text(text: str) -> Dict[torch.Tensor, torch.Tensor]:
        """
        Tokenize the words of the sentence and encode them
        """
        encoded_text = tokenizer.encode_plus(text, max_length=512,
                                             padding="max_length", truncation=True,
                                             return_attention_mask=True, return_tensors="pt")
        return {
            "input_ids": encoded_text["input_ids"].squeeze(),
            "attention_mask": encoded_text["attention_mask"].squeeze()
        }

    # Define the preprocessed clinical notes
    clinical_notes_tokenized = df_dk["clinical_notes"].apply(preprocess_text)

    # Split the data into training and test sets
    train_texts, val_texts = train_test_split(clinical_notes_tokenized, test_size=0.2, random_state=42)

    # Create a dataset class for our data
    class ClinicalNotesDataset(Dataset):
        def __init__(self, texts):
            self.texts = texts

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts.iloc[idx]
            return {
                "input_ids": text['input_ids'].squeeze(),
                "attention_mask": text['attention_mask'].squeeze(),
                "labels": text['input_ids'].squeeze()  # Use input_ids as labels for causal language modeling
            }

    # Create data loaders for training and validation
    train_dataset = ClinicalNotesDataset(train_texts)
    val_dataset = ClinicalNotesDataset(val_texts)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load the LLaMA model
    model = LlamaForCausalLM.from_pretrained(llm_model)

    # Ensure the model is on CPU
    device = torch.device("cpu")
    model.to(device)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=1e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        save_steps=10000,
        save_total_limit=2,
    )

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

    
if __name__ == "__main__":
    llama_test()
