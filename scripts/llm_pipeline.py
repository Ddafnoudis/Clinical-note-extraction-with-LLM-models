from pathlib import Path
from scripts.define_model import model_config
from scripts.tokize_input import tokenize_input
from scripts.query_tensor import query_tensor
from scripts.key_tensor import key_tensor
from scripts.embending_layer import embedding_layer
from scripts.self_attention import self_attention_qk
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
    q_per_token_rotated, freqs_cis = query_tensor(model=model, n_heads=n_heads, dim=dim, token=tokens, token_embeddings=token_embeddings, rope_theta=rope_theta)
    k_per_token_rotated = key_tensor(freqs_cis=freqs_cis, model=model, n_kv_heads=n_kv_heads, dim=dim, token=tokens, token_embeddings=token_embeddings)
    
    # Implement Self-Attention
    self_attention_qk(tokens=tokens,
                      q_per_token_rotated=q_per_token_rotated,
                      k_per_token_rotated=k_per_token_rotated)





if __name__ =="__main__":
    llm_pipeline()