# v2_kafka_12l

12-layer GPT-style transformer trained on Kafka + Dostoyevsky.

## Architecture

| Parameter | Value |
|-----------|-------|
| Layers | 12 |
| Embedding dim | 768 |
| Attention heads | 8 |
| FFN dim | 3072 (4× embed) |
| Activation | GELU |
| Normalization | Pre-norm (norm_first=True) |
| Batch-first | Yes |
| Vocab size | Derived from tokenizer at runtime |

**Output head:** `LayerNorm → Dropout → Linear(768 → vocab_size)`

## Corpus

Kafka + Dostoyevsky from Project Gutenberg (~13M words):
- Kafka: The Metamorphosis, The Trial
- Dostoyevsky: Notes from the Underground, The Gambler, Poor Folk, White Nights, Crime and Punishment

## Checkpoint naming

`transformer_dev_e{epoch}_b{batch}.pt`

## Training hyperparameters

| Parameter | Value |
|-----------|-------|
| Initial LR | 3e-3 |
| LR at epoch 10 | 1e-4 |
| Batch size | 64 |
| Dropout | 0.05 |
| Scheduler | Linear warmup (1000 steps) + cosine annealing |
| Grad clip | 1.0 |
| Label smoothing | 0.05 |
