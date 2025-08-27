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
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class ShakespeareLM(nn.Module):
    def __init__(self,
                 hidden_size: int = 1648,
                 embedding_dim: int = 768,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 vocab_size: int = 24943,
                 finetune_bert: bool = False,  # Kept for compatibility
                 num_heads: int = 8):
        super(ShakespeareLM, self).__init__()

        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)

        # Simple and clean: TransformerEncoder with causal masking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Simple output head for per-token prediction
        self.output_head = nn.Sequential(nn.LayerNorm(embedding_dim),
                                         nn.Dropout(dropout),
                                         nn.Linear(embedding_dim, hidden_size*2),
                                         nn.GELU(),
                                         nn.Dropout(dropout),
                                         nn.Linear(hidden_size*2, vocab_size))

    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for transformer decoder"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_tokens.shape

        # Embedding and positional encoding
        embeddings = self.embedding_layer(input_tokens)  # (batch_size, seq_len, embedding_dim)
        embeddings = embeddings.transpose(0, 1)  # (seq_len, batch_size, embedding_dim)
        embeddings = self.positional_encoding(embeddings)
        embeddings = embeddings.transpose(0, 1)  # (batch_size, seq_len, embedding_dim)

        # Create causal mask for autoregressive generation
        causal_mask = self.generate_square_subsequent_mask(seq_len).to(input_tokens.device)

        # Clean and simple forward pass
        transformer_out = self.transformer_encoder(
            embeddings,
            mask=causal_mask
        )

        # Generate logits for each position
        logits = self.output_head(transformer_out)  # (batch_size, seq_len, vocab_size)

        return logits


def top_p_sample_batch(prob_batch: torch.Tensor, p: float = 0.3):
    """
    Perform top-p (nucleus) sampling on a batch of probability tensors.

    Args:
        prob_batch (torch.Tensor): shape (batch_size, vocab_size), post-softmax probabilities.
        p (float): cumulative probability threshold for top-p sampling.

    Returns:
        List[int]: sampled token indices, length = batch_size
    """
    sampled_tokens = []

    for probs in prob_batch:  # probs shape = (vocab_size,)
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # Compute cumulative sum
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)

        # Find the cutoff index where cumulative probability > p
        cutoff_idx = torch.searchsorted(cumulative_probs, p).item() + 1

        # Keep only tokens within the top-p nucleus
        top_probs = sorted_probs[:cutoff_idx]
        top_indices = sorted_indices[:cutoff_idx]

        # Normalize top-p probabilities
        top_probs = top_probs / top_probs.sum()

        # Sample a token from the top-p distribution
        sampled_token = torch.multinomial(top_probs, num_samples=1).item()

        # Map back to original token index
        sampled_tokens.append(top_indices[sampled_token].item())

    return sampled_tokens