"""
Test Llama3 for feature extraction
"""
import ollama
from typing import List
from pandas import DataFrame

def model(df: DataFrame, keyword:str)-> List[str] :
    # Pull model
    try:
        ollama.pull("llama3")
        print("Model has been pulled")
    except Exception as error:
        print("Error", error)

    # Create an empty list
    cli_notes_list = []
    # Define the clinical notes
    clinical_notes = df["clinical_notes"]
    # Append notes to a list
    for notes in clinical_notes:
        cli_notes_list.append(notes)
    
    # Define the message for the model
    message = [
    {"role": "system", "content": "You are an excellent Danish physician in extracting relevant words from clinical notes"},
    {"role": "user", "content": "Given the following clinical notes:\n\n".join(cli_notes_list) + "\n\nFind clinically relevant words from the clinical list that are related to the keyword: " + keyword}
    ]
    
    # Response based on model and message
    try:
        response = ollama.chat(model='llama3', messages=message)
        return response['message']['content']
    # Capture an error  
    except Exception as error:
        print("Error", error)
        return "Error"
    

if __name__ == "__main__":
    model()