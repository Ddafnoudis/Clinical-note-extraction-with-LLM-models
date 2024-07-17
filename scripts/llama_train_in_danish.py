"""
Train llama3 on in Danish dataset
"""
import ollama
import torch
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from pandas import DataFrame
from sklearn.model_selection import train_test_split
# from transformers import AutoTokenizer, AutoModelForCausalLM


def train_llama3(df_dk: DataFrame):
    """
    Train the model with danish language text
    """
    # Pull model
    try:
        ollama.pull("llama3")
        print("Model has been pulled")
    except Exception as error:
        print("Error", error)

    # print(df_dk.shape)
    def train_test_text_data(df_dk):
        """
        Separate the clinical data to train and test
        """
        train_text, test_text = train_test_split(df_dk["clinical_notes"], test_size=0.2, random_state=42)

        return train_text, test_text
    
    train_text, test_text = train_test_text_data(df_dk)
    # print(f"Train_text:\n {train_text} \n\nTest_text: \n{test_text}")

    # Tokenize the train and test sets
    train_sentences = []
    test_sentences = []

    for text_tr in train_text:
        train_tokenized = nltk.word_tokenize(text_tr, language='danish')
        train_sentences.append(" ".join(train_tokenized))

    for text_te in test_text:
        test_tokenized = nltk.word_tokenize(text_te, language='danish')
        test_sentences.append(" ".join(test_tokenized))

    # Train the model
    for sentence in tqdm(train_sentences, desc="Training", unit="sentence"):
        prompt = f"Analyze this Danish clinical note and provide the symptoms of the patients with diabetes: {sentence}"
        response = ollama.chat(model='llama3', messages=[
            {'role': 'system', 'content': 'You are a Danish medical AI assistant. Analyze the given clinical note and provide the symptoms of the patients with diabetes. Use bullets.'},
            {'role': 'user', 'content': prompt}
        ])

    print("Training completed!")
    return response["message"]["content"]

    

if __name__ == "__main__":
    train_llama3()