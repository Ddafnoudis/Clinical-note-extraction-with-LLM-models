"""
Test Llama3 for feature extraction
"""
import ollama
from tqdm import tqdm
from typing import List
from pandas import DataFrame

def model(df: DataFrame, keyword: str, batch_size: int)-> List[str]:
    # Pull model
    try:
        ollama.pull("llama3")
        print("Model has been pulled! Ready for the English DataFrame!\n\n")
    except Exception as error:
        print("Error", error)

    # Define the clinical notes
    clinical_notes = df["clinical_notes"]
    # print(clinical_notes);exit()
    # Split clinical notes into batches (model's token limits)
    clinical_notes_batches = [clinical_notes[i:i + batch_size] for i in range(0, len(clinical_notes), batch_size)]
    # print(clinical_notes_batches);exit()
    # Create an empty list
    all_responses = []

    for batch in tqdm(clinical_notes_batches, desc="Braches"):
        # print(batch);exit()
        # Format the batches
        cli_notes_list = "\n\n".join(batch)
        # print(cli_notes_list);exit()
        # Create the message for the model
        message = [
        {"role": "user", 
         "content": f":What are the symptoms of diabetic patients from the clinical notes below?: \n\n{''.join(cli_notes_list)}"}
        ]
        # Response based on model and message
        try:
            response = ollama.chat(model='llama3', messages=message, strem=True,)
            all_responses.append(response["message"]["content"])
        # Capture an error  
        except Exception as error:
            print("Error", error)

    return all_responses


def model_danish(df_da: DataFrame, keyword: str, batch_size: int)-> List[str]:
    # Pull model
    try:
        ollama.pull("llama3")
        print("Model has been pulled! Ready for the Danish DataFrame!\n\n")
    except Exception as error:
        print("Error", error)

    # Create an empty list
    cli_notes_list = []
    # Define the clinical notes
    clinical_notes = df_da["clinical_notes"]
    # Append notes to a list
    for notes in clinical_notes:
        cli_notes_list.append(notes)
    
    # Define the message for the model
    message = [
    {"role": "system", "content": "You are a Danish physician"},
    {"role": "user", "content": "Given the following clinical notes:\n\n".join(cli_notes_list) + "\n\nWhat drugs are found in patients with " + keyword}
    ]

    # Response based on model and message
    try:
        response_danish = ollama.chat(model='llama3', messages=message)
        return response_danish['message']['content']
    # Capture an error  
    except Exception as error:
        print("Error", error)
        return "Error"


if __name__ == "__main__":
    model()
