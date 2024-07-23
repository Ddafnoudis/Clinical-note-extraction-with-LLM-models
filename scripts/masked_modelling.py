"""
Train Llama3-8B into masked language model tasks.
Perform a performance metric calulation
"""
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
import numpy as np
from typing import List, Any, Dict
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, LlamaTokenizer, LlamaConfig
from torch.utils.data import Dataset, DataLoader


def mask_words(train_text: List[str], test_text: List[str], llm_model: str) -> Any:
    """
    """
    # Define the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_model)

    def preprocess(train_text: List[str])-> Dict[List[List[int]], List[List[int]]] :
        """"
        A function that tokenize the words of clinical train text
        """
        return tokenizer([" ".join(text) for text in train_text])
    
    # Define the tokenized words 
    tokenized_train_text = preprocess(train_text)
    # print(tokenized_train_text);exit()

    # Create a custom dataset class for our clinical text data
    class ClinicalTextDataset(Dataset):
        def __init__(self, data, tokenizer, max_len):
            self.data = data
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            text = self.data[idx]
            encoding = self.tokenizer.encode_plus(
                text,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': encoding['input_ids'].flatten()
            }

    # Set the maximum sequence length
    max_len = 512

    # Create the tokenizer and model
    config = LlamaConfig.from_pretrained(llm_model)
    model = AutoModelForCausalLM.from_pretrained(llm_model, config=config)

    # Set the device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = DataParallel(model)
    
    model.to(device)

    # Create the dataset and data loader
    dataset = ClinicalTextDataset(tokenized_train_text, tokenizer, max_len)
    batch_size = 4
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Set the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Set number of worker processes for data loading
    num_workers = mp.cpu_count()
    torch.cuda.empty_cache()
    # Set the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

     # Create the dataset and data loader with multiple workers
    dataset = ClinicalTextDataset(tokenized_train_text, tokenizer, max_len)
    batch_size = 4
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    torch.cuda.empty_cache()
    # Train the model
    for epoch in range(5):  # train for 5 epochs
        model.train()
        total_loss = 0
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            torch.cuda.empty_cache()

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}')

        model.eval()
        


if __name__ == "__main__":
    mask_words()
