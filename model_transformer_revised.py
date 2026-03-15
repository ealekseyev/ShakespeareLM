import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Fix: store as (1, max_len, embedding_dim) for batch-first tensors.
        # Original stored as (max_len, 1, embedding_dim) and relied on seq-first
        # transposes in forward() that were mismatched with batch_first=True layers.
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, embedding_dim)
        return x + self.pe[:, :x.size(1), :]


class ShakespeareLM(nn.Module):
    def __init__(self,
                 hidden_size: int = 1648,
                 embedding_dim: int = 768,
                 num_layers: int = 7,
                 dropout: float = 0.2,
                 vocab_size: int = 24943,
                 pad_token_id: int = 24942,
                 finetune_bert: bool = False,  # Kept for checkpoint compatibility
                 num_heads: int = 8):
        super(ShakespeareLM, self).__init__()

        self.pad_token_id = pad_token_id
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        )

        # Fix: slim output head — direct projection without large intermediate expansion.
        # Original head (768→3296→vocab) had ~84M params (more than the transformer body),
        # dominating gradient updates and slowing transformer layer learning.
        self.output_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, vocab_size)
        )

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Causal mask using a large finite negative value.
        Avoids NaN gradients that float('-inf') can produce with pre-norm (norm_first=True)
        when unnormalized residual values are large early in training."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, -1e4)
        return mask

    def forward(self, input_tokens: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            input_tokens: (batch, seq_len) token indices
            padding_mask: optional (batch, seq_len) bool tensor where True = padding position.
                          Auto-derived from _PAD_TOKEN_ID if not provided.
        """
        batch_size, seq_len = input_tokens.shape

        # Embedding + positional encoding — batch-first throughout, no transpose needed.
        embeddings = self.embedding_layer(input_tokens)       # (batch, seq_len, embedding_dim)
        embeddings = self.positional_encoding(embeddings)     # (batch, seq_len, embedding_dim)

        causal_mask = self.generate_square_subsequent_mask(seq_len, input_tokens.device)

        # Fix: apply padding mask so padding tokens don't corrupt real-token gradients.
        if padding_mask is None:
            padding_mask = (input_tokens == self.pad_token_id)  # (batch, seq_len), True = ignore

        # Convert bool padding mask to float to match causal_mask dtype.
        # PyTorch 2.x requires mask and src_key_padding_mask to be the same type.
        # Bool True = ignore → float -1e4 (added to attention logits, effectively -inf).
        padding_mask = padding_mask.float().masked_fill(padding_mask, -1e4)

        transformer_out = self.transformer_encoder(
            embeddings,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )

        logits = self.output_head(transformer_out)            # (batch, seq_len, vocab_size)
        return logits


def top_p_sample_batch(prob_batch: torch.Tensor, p: float = 0.3):
    """
    Top-p (nucleus) sampling on a batch of probability tensors.

    Fix: original used searchsorted(cumulative_probs, p) + 1 which could clip too
    aggressively. Standard implementation: keep tokens where the cumulative probability
    BEFORE this token is below p, ensuring the token that first pushes past p is included.

    Args:
        prob_batch: (batch_size, vocab_size) post-softmax probabilities.
        p: cumulative probability threshold.

    Returns:
        List[int]: sampled token indices, length = batch_size
    """
    sampled_tokens = []

    for probs in prob_batch:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)

        # Shift cumulative sum right by one: nucleus includes every token up to and
        # including the one that first pushes cumulative probability past p.
        shifted = torch.cat([torch.zeros(1, device=probs.device), cumulative_probs[:-1]])
        nucleus_mask = shifted < p  # (vocab_size,) bool

        top_probs = sorted_probs[nucleus_mask]
        top_indices = sorted_indices[nucleus_mask]

        # Guarantee at least one token (handles edge case where p is extremely small)
        if top_probs.numel() == 0:
            top_probs = sorted_probs[:1]
            top_indices = sorted_indices[:1]

        top_probs = top_probs / top_probs.sum()
        sampled_token = torch.multinomial(top_probs, num_samples=1).item()
        sampled_tokens.append(top_indices[sampled_token].item())

    return sampled_tokens
