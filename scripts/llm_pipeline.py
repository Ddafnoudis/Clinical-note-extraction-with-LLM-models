from pathlib import Path
from scripts.define_model import model_config
from scripts.tokize_input import tokenize_input
from scripts.attention_heads import query_key_value
from scripts.embending_layer import embedding_layer
from scripts.param_config import param_configuration
from scripts.preprocessing_text import preprocessing_text
from scripts.tokenizer_preprocess import preprocess_tokenizer



def llm_pipeline(df_dk_path: Path, tokenizer_model: Path,
                 model: Path, params_config: Path):
    """
    """
    # Configure the model
    tokenizer_model, model, params_config = model_config(tokenizer=tokenizer_model, model=model, params=params_config)
    dim, n_layers, n_heads, n_kv_heads, vocab_size, multiple_of, ffn_dim_multiplier, norm_eps, rope_theta = param_configuration(config=params_config)
    tokenizer = preprocess_tokenizer(tokenizer_model=tokenizer_model)
    # Preprocess the clinical text data
    train_text, test_text = preprocessing_text(df_danish_path=df_dk_path)
    # Tokenize the tokens
    tokens = tokenize_input(tokenizer=tokenizer, train_text=train_text)
    # Define the embedding tokens and normalize them
    token_embeddings = embedding_layer(model=model, vocab_size=vocab_size, dim=dim, tokens=tokens, norm_eps=norm_eps)
    #
    query_key_value(model=model, n_heads=n_heads, dim=dim, token=tokens, token_embeddings=token_embeddings)



if __name__ =="__main__":
    llm_pipeline()