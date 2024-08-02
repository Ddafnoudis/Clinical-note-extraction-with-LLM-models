"""
"""
from pathlib import Path
from scripts.define_model import model_config
from scripts.tokize_input import tokenize_input
from scripts.query_tensor import query_tensor
from scripts.key_tensor import key_tensor
from scripts.embending_layer import embedding_layer
from scripts.self_attention import self_attention_qkv
from scripts.param_config import param_configuration
from scripts.final import final_all_layers_merged
from scripts.preprocessing_text import preprocessing_text
from scripts.tokenizer_preprocess import preprocess_tokenizer
from scripts.multi_head_attention import multi_head_attention
from scripts.swiglu_act_funtion import swiglu_activation_function


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
    token_embeddings, token_embeddings_unnormal = embedding_layer(model=model, vocab_size=vocab_size, dim=dim, tokens=tokens, norm_eps=norm_eps)
    #
    q_layer_0, q_per_token_rotated, freqs_cis = query_tensor(model=model, n_heads=n_heads, dim=dim, token=tokens, token_embeddings=token_embeddings, rope_theta=rope_theta)
    k_layer_0, k_per_token_rotated = key_tensor(freqs_cis=freqs_cis, 
                                                model=model, 
                                                n_kv_heads=n_kv_heads, 
                                                dim=dim, 
                                                token_embeddings=token_embeddings)
    
    # Implement Self-Attention
    v_layer_0 = self_attention_qkv(model=model,
                       n_kv_heads=n_kv_heads,
                       token_embeddings=token_embeddings,
                       dim=dim,
                       tokens=tokens,
                       q_per_token_rotated=q_per_token_rotated,
                       k_per_token_rotated=k_per_token_rotated)
    
    embedding_after_edit_normalized = multi_head_attention(n_heads=n_heads, q_layer_0=q_layer_0, 
                        k_layer_0=k_layer_0, v_layer_0=v_layer_0, 
                        tokens=tokens, token_embeddings=token_embeddings, 
                        freqs_cis=freqs_cis,
                        norm_eps=norm_eps,
                        model=model,
                        token_embeddings_unnormal=token_embeddings_unnormal)

    output_after_feedforward = swiglu_activation_function(embedding_after_edit_normalized=embedding_after_edit_normalized,
                               model=model)
    
    final_all_layers_merged(token_embeddings_unnormal=token_embeddings_unnormal, 
                            model=model, n_layers=n_layers, n_heads=n_heads,
                            dim=dim, n_kv_heads=n_kv_heads, norm_eps=norm_eps,
                            freqs_cis=freqs_cis)
    



if __name__ =="__main__":
    llm_pipeline()
