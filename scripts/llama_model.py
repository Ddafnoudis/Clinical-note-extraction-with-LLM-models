"""
Test Llama3 in prompt that extracts symptoms based on keyword in
Danish clinical notes
Steps:  1) Pulling the model
        2) Create batches of clinical notes
        3) Prompt Engineering
        4) Ollama chat
        5) Save results
"""
import sys
import time
import ollama
import random
from tqdm import tqdm
from pandas import DataFrame
from typing import List, Tuple


# Reconfigure the standard output stream to use UTF-8 encoding.
sys.stdout.reconfigure(encoding="utf-8")


def model_danish(df_dk: DataFrame, keyword: str, batch_size: int)-> List[Tuple[str]]:
    """
    Extract symptoms from danish clinical notes using Ollama library 
    """
    # Pull model
    try:
        ollama.pull("llama3")
        print("Model has been pulled")
    except Exception as error:
        print("Error", error)

    # Define the clinical notes
    clinical_notes = df_dk["clinical_notes"]

    # Split the clinical_notes DataFrame into batches of size batch_size.
    batches = [clinical_notes[i:i + batch_size] for i in range(0, len(clinical_notes), batch_size)]

    # Create a list to store the response of the model
    response_list = []

    # Iterate over batches
    for batch in tqdm(batches, desc="Processing batches"):
        # Construct the message for the current batch
        messages = [
            {
                "role": "user",
                "content": (
                    f"Given the following clinical notes, please identify the patients diagnosed with {keyword} "
                    f"in the text. Find their symptoms and write only the symptoms using comma (e.g Patient Index 1: Symptom1, symptom2 ..., symptom) else if the patient in: {batch} do not has {keyword}, write the patient index and None and the disease and nothing else about them (e.g Patient Index 1: None):\n" + "\n\n".join(batch)
                )
            }
        ]

        try:
            # Define the response of the model
            response = ollama.chat(model='llama3', messages=messages)
            # Define the response of the moder
            response_content = response["message"]["content"]
            # Append to the list
            response_list.append(response_content)
            # Save the response list to a file
            with open('response_list.txt', 'w', encoding='utf-8') as file:
                for response in response_list:
                    file.write(response + '\n\n')

            # Wait before use the chat again
            time.sleep(random.randint(1, 3))
        # Create an exception if error occured during response
        except Exception as error:
            print(f"{error}")


if __name__ == "__main__":
    model_danish()
