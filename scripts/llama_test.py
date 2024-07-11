import torch
from tqdm import tqdm
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM


def llama_test(df_dk: DataFrame, llm_model: str):

    # Preprocess the text data
    tokenizer = AutoTokenizer.from_pretrained(
        llm_model,
        padding=True,
        padding_side="left",
        pad_token="[PAD]",
        add_eos_token=True,
        add_bos_token=True,
        )

    tokenizer.pad_token = tokenizer.eos_token

    def preprocess_text(text):
        return tokenizer.encode_plus(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

    df_dk["clinical_notes"] = df_dk["clinical_notes"].apply(preprocess_text)
    # print(df_dk["clinical_notes"]);exit()
    # Split the data into training and test sets
    train_texts, val_texts = train_test_split(df_dk["clinical_notes"], test_size=0.2, random_state=42)

    # Create a dataset class for our data
    class ClinicalNotesDataset(torch.utils.data.Dataset):
        def __init__(self, texts):
            self.texts = texts

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts.iloc[idx]
            # input_ids = text["input_ids"].squeeze()
            # attention_mask = text["attention_mask"].squeeze()
            return {
                "input_ids": text['input_ids'].squeeze(),
                "attention_mask": text['attention_mask'].squeeze()
                }

    # Create data loaders for training and validation
    train_dataset = ClinicalNotesDataset(train_texts)
    # print(train_dataset);exit()
    val_dataset = ClinicalNotesDataset(val_texts)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    # print(train_loader);exit()
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Load the LLaMA model
    model = AutoModelForCausalLM.from_pretrained(
        llm_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(5):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = input_ids.clone()
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        print(f"Epoch {epoch+1}, Val Loss: {total_loss / len(val_loader)}")


if __name__ == "__main__":
    llama_test()