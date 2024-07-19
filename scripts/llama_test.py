"""
Training an Large Language Model (LLM) in Danish clinical text data
"""
import torch
torch.cuda.empty_cache()
import gc
gc.collect()
from pathlib import Path
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    AutoModelForCausalLM,
    )


from trl import SFTTrainer
from tqdm import tqdm
from typing import Dict
from pandas import DataFrame
from sklearn.model_selection import train_test_split


def llama_test(df_dk: DataFrame, llm_model: Path, model_path: Path, tokenizer_path: Path, batch_size: int, new_model: str, num_epochs: int):
    """
    Train the Llama3 model in danish clinical text data
    """
    device = torch.device("cpu")
    # Load base model as set it to the current device
    model = AutoModelForCausalLM.from_pretrained(llm_model)
    model = model.to(device)

    # don't use the cache
    model.config.use_cache = False

    # Define the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_model)
    # tokenizer.pad_token = tokenizer.eos_token

    # Input_ids and attention_mask
    def preprocess_text(text: str) -> str: #Dict[torch.Tensor, torch.Tensor]:
        """
        Tokenize the words of the sentence and encode them
        """
        # encoded_text = tokenizer.encode_plus(text, max_length=512,
        #                                      padding="max_length", truncation=True,
        #                                      return_attention_mask=True, return_tensors="pt")
        # return {
        #     "input_ids": encoded_text["input_ids"].squeeze(),
        #     "attention_mask": encoded_text["attention_mask"].squeeze()
        # }

        return text
    # Define the preprocessed clinical notes
    clinical_notes_preprocessed= df_dk["clinical_notes"].apply(preprocess_text)

    # Convert to list
    clinical_notes_list = clinical_notes_preprocessed.tolist()  

    # Split the data into training and test sets
    train_texts, val_texts = train_test_split(clinical_notes_list, test_size=0.2, random_state=42)
    
    # Create a Dataset from the dictionary
    train_dataset = Dataset.from_dict({"text": train_texts})
    #print(train_dataset);exit()
    val_dataset = Dataset.from_dict({"text": val_texts})

    train_df_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # for batch in train_df_loader:
    #     print(batch)
    val_df_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        use_cpu=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        optim="adamw_torch",
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=0,
        fp16=False,  # Unable mixed precision training
        bf16=False,  # Disable bfloat16
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        gradient_checkpointing=True,
    )

    torch.cuda.empty_cache()

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        compute_metrics=None,
        train_dataset=train_dataset,
        dataset_text_field="text",
        )

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(num_epochs):
        model.train()
        for batch in train_df_loader:
            inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
    # Train model
    # trainer.train()

    # Save trained model
    trainer.model.save_pretrained(new_model)

    
if __name__ == "__main__":
    llama_test()
