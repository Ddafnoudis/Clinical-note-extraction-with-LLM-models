# """
# Merges all the layers of a neural network model by applying attention and 
# feedforward operations to the token embeddings
# """
# import torch
# from tqdm import tqdm
# from typing import Dict


# def final_all_layers_merged(token_embeddings_unnormal: torch.Tensor, 
#                             model: Dict, n_layers: int, n_heads: int,
#                             dim: int, n_kv_heads: int, norm_eps: int,
#                             freqs_cis: torch.Tensor)-> torch.Tensor:
#     """
#     """
#     # Initialize final embedding with unnormalized token embeddings
#     final_embedding = token_embeddings_unnormal

#     # Calculating RMSNorm
#     def rms_norm(tensor, norm_weights):
#         """
#         Applie root-mean-square (RMS) normalization
#         """
#         # Calculate the mean of the square of tensor values along the last dimension
#         squared_mean: torch.Tensor = tensor.pow(2).mean(-1, keepdim=True)
#         # Add a small value to avoid division by zero
#         normalized: torch.Tensor = torch.rsqrt(squared_mean + norm_eps)
    
#         # Multiply normalized tensor by the provided normalization weights
#         return (tensor * normalized) * norm_weights

#     # Iterate through each layer
#     for layer in tqdm(range(n_layers), desc="Progress", total=n_layers, unit="layer"):
#         # Initialize list to store QKV attentions for each head
#         qkv_attention_store = []
        
#         # Normalize the final embedding using root mean square normalization and weights from the current layer
#         layer_embedding_norm = rms_norm(final_embedding, 
#                                         norm_weights=model[f"layers.{layer}.attention_norm.weight"])
        
#         # Retrieve query, key, value, and output weights for the attention mechanism of the current layer
#         q_layer = model[f"layers.{layer}.attention.wq.weight"]
#         q_layer = q_layer.view(n_heads, q_layer.shape[0] // n_heads, dim)
#         k_layer = model[f"layers.{layer}.attention.wk.weight"]
#         k_layer = k_layer.view(n_kv_heads, k_layer.shape[0] // n_kv_heads, dim)
#         v_layer = model[f"layers.{layer}.attention.wv.weight"]
#         v_layer = v_layer.view(n_kv_heads, v_layer.shape[0] // n_kv_heads, dim)
#         w_layer = model[f"layers.{layer}.attention.wo.weight"]
        
#         # Iterate through each head
#         for head in range(n_heads):
#             # Extract query, key, and value weights for the current head
#             q_layer_head = q_layer[head]
#             k_layer_head = k_layer[head//4]  # Key weights are shared across 4 heads
#             v_layer_head = v_layer[head//4]  # Value weights are shared across 4 heads
            
#             # Calculate query per token by matrix multiplication
#             q_per_token = torch.matmul(layer_embedding_norm, q_layer_head.T)
            
#             # Calculate key per token by matrix multiplication
#             k_per_token = torch.matmul(layer_embedding_norm, k_layer_head.T)
            
#             # Calculate value per token by matrix multiplication
#             v_per_token = torch.matmul(layer_embedding_norm, v_layer_head.T)
            
#             # Split query per token into pairs and rotate them
#             q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
#             q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
#             q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis)
#             q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)
            
#             # Split key per token into pairs and rotate them
#             k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
#             k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
#             k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis)
#             k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)
            
#             # Calculate query-key dot products per token
#             qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T) / (128) ** 0.5
            
#             # Create a mask tensor filled with negative infinity values
#             mask = torch.full((len(token_embeddings_unnormal), len(token_embeddings_unnormal)), float("-inf"))
#             # Set upper triangular part of the mask tensor to negative infinity
#             mask = torch.triu(mask, diagonal=1)
#             # print(mask)
#             # Add the mask to the query-key dot products per token
#             qk_per_token_after_masking = qk_per_token + mask
#             # print(qk_per_token_after_masking)
#             # Apply softmax along the second dimension after masking
#             qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
            
#             # Calculate QKV attention by matrix multiplication
#             qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
            
#             # Store QKV attention for the current head
#             qkv_attention_store.append(qkv_attention)
        
#         # Concatenate QKV attentions from all heads along the last dimension
#         stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
        
#         # Calculate embedding delta by matrix multiplication with the output weight
#         embedding_delta = torch.matmul(stacked_qkv_attention, w_layer.T)
        
#         # Add the embedding delta to the current embedding to get the edited embedding
#         embedding_after_edit = final_embedding + embedding_delta
        
#         # Normalize the edited embedding using root mean square normalization and weights from the current layer
#         embedding_after_edit_normalized = rms_norm(embedding_after_edit, 
#                                                    norm_weights=model[f"layers.{layer}.ffn_norm.weight"])
        
#         # Retrieve weights for the feedforward layer
#         w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
#         w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
#         w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
        
#         # Perform operations for the feedforward layer
#         output_after_feedforward = torch.matmul(torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(embedding_after_edit_normalized, w3.T), w2.T)
        
#         # Update the final embedding with the edited embedding plus the output from the feedforward layer
#         final_embedding = embedding_after_edit + output_after_feedforward

#     # Normalize the final embedding using root mean square normalization and provided weights
#     final_embedding = rms_norm(final_embedding, 
#                                norm_weights=model["norm.weight"])

#     # Print the shape of the resulting normalized final embedding
#     print(f"Shape of the final embedding: {final_embedding.shape}")

#     return final_embedding


# if __name__=="__main__":
#     final_all_layers_merged()


import torch
from torch import nn
from tqdm import tqdm
from typing import Dict

class LLama3_1(nn.Module):
    def __init__(self, model: Dict, n_layers: int, n_heads: int,
                 dim: int, n_kv_heads: int, norm_eps: float, freqs_cis: torch.Tensor):
        """
        Initialize the LLama3_1 class with the necessary parameters.
        """
        super(LLama3_1, self).__init__()
        self.model = model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = dim
        self.n_kv_heads = n_kv_heads
        self.norm_eps = norm_eps
        self.freqs_cis = freqs_cis
        
        # Correctly initialize the token embedding weights across layers
        self.tok_embeddings_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(model["tok_embeddings.weight"])) for _ in range(n_layers)
        ])
        
        # Initialize other weights
        self.attention_norm_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(model[f"layers.{i}.attention_norm.weight"])) for i in range(n_layers)
        ])
        self.attention_wq_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(model[f"layers.{i}.attention.wq.weight"])) for i in range(n_layers)
        ])
        self.attention_wk_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(model[f"layers.{i}.attention.wk.weight"])) for i in range(n_layers)
        ])
        self.attention_wv_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(model[f"layers.{i}.attention.wv.weight"])) for i in range(n_layers)
        ])
        self.attention_wo_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(model[f"layers.{i}.attention.wo.weight"])) for i in range(n_layers)
        ])
        self.ffn_norm_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(model[f"layers.{i}.ffn_norm.weight"])) for i in range(n_layers)
        ])
        self.ffn_w1_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(model[f"layers.{i}.feed_forward.w1.weight"])) for i in range(n_layers)
        ])
        self.ffn_w2_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(model[f"layers.{i}.feed_forward.w2.weight"])) for i in range(n_layers)
        ])
        self.ffn_w3_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(model[f"layers.{i}.feed_forward.w3.weight"])) for i in range(n_layers)
        ])


    def rms_norm(self, tensor: torch.Tensor, norm_weights: torch.Tensor) -> torch.Tensor:
        """
        Applies root-mean-square (RMS) normalization to the given tensor.
        """
        squared_mean = tensor.float().pow(2).mean(-1, keepdim=True)
        normalized = torch.rsqrt(squared_mean + self.norm_eps)
        return (tensor.float() * normalized) * norm_weights

    def apply_attention(self, layer_embedding_norm: torch.Tensor, layer: int) -> torch.Tensor:
        """
        Applies the attention mechanism for a single layer.
        """
        qkv_attention_store = []
        q_layer = self.attention_wq_weights[layer].view(self.n_heads, -1, self.dim)
        k_layer = self.attention_wk_weights[layer].view(self.n_kv_heads, -1, self.dim)
        v_layer = self.attention_wv_weights[layer].view(self.n_kv_heads, -1, self.dim)
        w_layer = self.attention_wo_weights[layer]

        for head in range(self.n_heads):
            q_layer_head = q_layer[head]
            k_layer_head = k_layer[head // 4]
            v_layer_head = v_layer[head // 4]

            q_per_token = torch.matmul(layer_embedding_norm, q_layer_head.T)
            k_per_token = torch.matmul(layer_embedding_norm, k_layer_head.T)
            v_per_token = torch.matmul(layer_embedding_norm, v_layer_head.T)

            q_per_token_rotated = self.rotate_pairs(q_per_token)
            k_per_token_rotated = self.rotate_pairs(k_per_token)

            qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T) / (128) ** 0.5

            mask = torch.full((len(layer_embedding_norm), len(layer_embedding_norm)), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            qk_per_token_after_masking = qk_per_token + mask

            qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
            qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)

            qkv_attention_store.append(qkv_attention)

        stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
        embedding_delta = torch.matmul(stacked_qkv_attention, w_layer.T)

        return embedding_delta

    def rotate_pairs(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Splits tensor into pairs and applies rotation using precomputed frequencies.
        """
        tensor_split_into_pairs = tensor.float().view(tensor.shape[0], -1, 2)
        tensor_as_complex_numbers = torch.view_as_complex(tensor_split_into_pairs)
        tensor_split_into_pairs_rotated = torch.view_as_real(tensor_as_complex_numbers * self.freqs_cis)
        return tensor_split_into_pairs_rotated.view(tensor.shape)

    def apply_feedforward(self, embedding: torch.Tensor, layer: int) -> torch.Tensor:
        """
        Applies the feedforward network for a single layer.
        """
        embedding_after_edit_normalized = self.rms_norm(embedding, self.ffn_norm_weights[layer])

        w1 = self.ffn_w1_weights[layer]
        w2 = self.ffn_w2_weights[layer]
        w3 = self.ffn_w3_weights[layer]

        output_after_feedforward = torch.matmul(
            torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(embedding_after_edit_normalized, w3.T), w2.T
        )
        return output_after_feedforward

    def final_all_layers_merged(self, token_embeddings_unnormal: torch.Tensor) -> torch.Tensor:
        """
        Merges all the layers of the neural network model by applying attention and feedforward operations
        to the token embeddings.
        """
        final_embedding = token_embeddings_unnormal

        for layer in tqdm(range(self.n_layers), desc="Progress", total=self.n_layers, unit="layer"):
            layer_embedding_norm = self.rms_norm(final_embedding, self.model[f"layers.{layer}.attention_norm.weight"])

            attention_embedding_delta = self.apply_attention(layer_embedding_norm, layer)
            embedding_after_attention = final_embedding + attention_embedding_delta

            feedforward_embedding_delta = self.apply_feedforward(embedding_after_attention, layer)
            final_embedding = embedding_after_attention + feedforward_embedding_delta

        return final_embedding

    def forward(self, token_embeddings_unnormal: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass through the model.
        """
        return self.final_all_layers_merged(token_embeddings_unnormal=token_embeddings_unnormal)


# Example usage
if __name__ == "__main__":
    LLama3_1()
    