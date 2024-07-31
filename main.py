"""
A pipeline for LLama-3.1-8B models training in Danish clinical text data
"""
# Import libraries and modules
from pathlib import Path
from scripts.define_model import model_config
from scripts.tokize_input import tokenize_input
from scripts.attention_heads import query_key_value
from scripts.embending_layer import embedding_layer
from scripts.param_config import param_configuration
from scripts.preprocessing_text import preprocessing_text
from scripts.tokenizer_preprocess import preprocess_tokenizer

# Define the path of the packages
DF_DK = Path("~/Desktop/Clinical-note-extraction-with-LLM-models/dataset_notes/clin_note_danish_df.tsv").expanduser()

# Define the paths of the model and the model's tokenizer
TOKENIZER = "llama3_1/Meta-Llama-3.1-8B-Instruct/tokenizer.model"
LLM_MODEL = "llama3_1/Meta-Llama-3.1-8B-Instruct/consolidated.00.pth"
PARAMS = "llama3_1/Meta-Llama-3.1-8B-Instruct/params.json"


def main():
    # Configure the model
    tokenizer_model, model, params_config = model_config(tokenizer=TOKENIZER, model=LLM_MODEL, params=PARAMS)
    dim, n_layers, n_heads, n_kv_heads, vocab_size, multiple_of, ffn_dim_multiplier, norm_eps, rope_theta = param_configuration(config=params_config)
    tokenizer = preprocess_tokenizer(tokenizer_model=tokenizer_model)
    # Preprocess the clinical text data
    train_text, test_text = preprocessing_text(df_danish_path=DF_DK)
    # Tokenize the tokens
    tokens = tokenize_input(tokenizer=tokenizer, train_text=train_text)
    # Define the embedding tokens and normalize them
    token_embeddings = embedding_layer(model=model, vocab_size=vocab_size, dim=dim, tokens=tokens, norm_eps=norm_eps)

    #
    query_key_value(model=model, n_heads=n_heads, dim=dim, token=tokens, token_embeddings=token_embeddings)


if __name__=="__main__":
    main()
