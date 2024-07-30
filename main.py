"""
Testing Llama3 model for extraction of relevant words
"""
# Import libraries and modules
from pathlib import Path
from scripts.define_model import model_config
from scripts.tokize_input import tokenize_input
from scripts.param_config import param_configuration
from scripts.preprocessing_text import preprocessing_text
from scripts.tokenizer_preprocess import preprocess_tokenizer

# Define keyword and paths
# KEYWORD = input("Enter the word that you are interested in: ")

# Define the path of the packages
DF_DK = Path("~/Desktop/Clinical-note-extraction-with-LLM-models/dataset_notes/clin_note_danish_df.tsv").expanduser()

# Define the paths of the model and the model's tokenizer
TOKENIZER = "llama3_1/Meta-Llama-3.1-8B-Instruct/tokenizer.model"
LLM_MODEL = "llama3_1/Meta-Llama-3.1-8B-Instruct/consolidated.00.pth"
PARAMS = "llama3_1/Meta-Llama-3.1-8B-Instruct/params.json"


def main():
    """
    """
    # Configure the model
    tokenizer_model, model, params_config = model_config(tokenizer=TOKENIZER, model=LLM_MODEL, params=PARAMS)
    dim, n_layers, n_heads, n_kv_heads, vocab_size, multiple_of, ffn_dim_multiplier, norm_eps, rope_theta = param_configuration(config=params_config)
    tokenizer = preprocess_tokenizer(tokenizer_model=tokenizer_model)
    # Preprocess the clinical text data
    train_text, test_text = preprocessing_text(df_danish_path=DF_DK)
    tokenize_input(tokenizer=tokenizer, train_text=train_text)


if __name__=="__main__":
    main()
