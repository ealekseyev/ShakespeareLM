# Shakespeare Language Model

A PyTorch implementation of transformer-based language models trained on Shakespeare's complete works to generate Shakespeare-esque text.

## Overview

This project explores autoregressive text generation using the Transformer encoder architecture with casual masking to mimic a standalone decoder. The model learns to predict the next word in a sequence, generating Shakespeare-like passages with grammar, punctuation, etc.

## Key Features

- **Word-Level Tokenization**: Uses regex-based tokenization preserving punctuation as separate tokens
- **Advanced Dataset Handling**: Generates 100K+ diverse training sequences per epoch with custom file sampling
- **Interactive Text Generation**: Real-time word-by-word output with proper formatting rules
- **Smart Text Formatting**: Automatic capitalization, contraction handling, and punctuation spacing, and sampling from model's raw logits based on previously generated context.

## Architecture Details

**Transformer Model** (`model_transformer.py`)
- GPT-style decoder-only architecture using TransformerEncoder with causal masking
- 768-dimensional embeddings with 1648-dimensional feedforward layers
- 3-layer feedforward model head
- Positional encoding and pre-layer normalization for training stability
- Multi-layer output head for robust next-token prediction

**Training** (`train.py`)
- Cosine annealing learning rate scheduling with linear warmup
- Gradient clipping and weight decay for stable convergence
- Sequence-to-sequence training with proper target shifting
- Regular checkpointing and evaluation metrics

**Dataset Processing** (`dataset.py`)
- Memory-efficient file-based sampling from large text corpus
- Variable sequence lengths (3-100 tokens) with configurable overlap
- Dynamic train/test splitting with diverse sampling strategies

## Technical Implementation

- **Framework**: PyTorch & custom dataset class
- **Training**: Adam optimizer with label smoothing and CE loss
- **Generation**: Top-p (nucleus) sampling with some additional rule-based filtering

## Project Structure

```
dev/
├── model_transformer.py    # Transformer architecture
├── train.py               # Training loop and hyperparameters
├── generate.py            # Interactive text generation
├── dataset.py             # Custom PyTorch dataset
└── tokenizer.py           # Word-level tokenization functions
```