# ShakespeareLM — Model Iterations

## V1 — Initial Shakespeare Model
**Branch:** `main` (first commit, Aug 2025)
**File:** `model_transformer.py`

### Architecture
- **Transformer:** 5-layer `nn.TransformerEncoder`, 8 heads, 768-dim embeddings, 1648-dim FFN
- **Output head (heavy, 3-component):** `LayerNorm → Dropout → Linear(768→3296) → GELU → Dropout → Linear(3296→vocab)`
  - The head alone was ~84M parameters — larger than the transformer body — which dominated gradient updates and slowed transformer layer learning
- **Dataset:** Shakespeare (`shakespeare_cleaned.txt`)

### Known Bugs
- **Positional encoding mismatch:** PE tensor was stored as `(max_len, 1, emb)` (seq-first) while the transformer ran `batch_first=True`. Forward pass applied manual `.transpose(0,1)` workarounds that were misaligned, corrupting positional information.
- **Causal mask used `float('-inf')`:** With `norm_first=True` (pre-norm), large early residual values caused NaN gradients when `-inf` was added to attention logits.
- **top_p off-by-one:** Used `searchsorted(cumulative_probs, p) + 1` which could over-clip the nucleus, skewing sampling toward only the top token.

### Performance
- **Min loss:** ~5.0 (plateaued, no further improvement)
- **Behavior:** Dominated by high-frequency token predictions — punctuation, pronouns, common function words. Rarely produced meaningful content words.

---

## V2 — Kafka/Dostoyevsky Model (Architectural Overhaul)
**Branches:** `kafka_deep`, `kafka_dostoyevsky`
**File:** `model_transformer_revised.py` (introduced at commit `3dcd027`)

### Architecture
- **Transformer:** 7 layers, 8 heads, 768-dim, 1648-dim FFN
- **Output head (slim, 1-layer):** `LayerNorm → Dropout → Linear(768→vocab)` — eliminated the large intermediate expansion
- **Dataset:** Kafka + Dostoyevsky (`kafka_dostoyevsky.txt`) — smaller, more modern vocabulary (~15,000 tokens vs ~24,000 for Shakespeare)

### Bug Fixes vs V1
- PE tensor reshaped to `(1, max_len, emb)` for clean batch-first broadcasting — no more transposes needed
- Causal mask switched to `-1e4` (finite) to avoid NaN gradients under pre-norm
- top_p fixed to use shifted cumulative sum (standard nucleus sampling)
- Padding mask added so pad tokens don't corrupt real-token gradients

### Performance
- **Min loss:** ~1.5
- **Behavior:** Noticeably more coherent. The slimmer head allowed the transformer layers to learn more effectively. The Kafka/Dostoyevsky corpus (shorter sentences, simpler syntax) was more learnable at this model size.

---

## V3 — Deeper Shakespeare Model (Current)
**Branch:** `deeper_transformer`
**File:** `model_transformer_revised.py` (same file, updated defaults)

### Architecture
- **Transformer:** 12 layers, 8 heads, 768-dim embeddings, **3072-dim FFN** (4× V2's 1648-dim internal FFN)
- **Output head:** Same slim 1-layer head as V2 — `LayerNorm → Dropout → Linear(768→vocab)`
- **Dataset:** Shakespeare (`shakespeare_cleaned.txt`) — larger, archaic vocabulary

### Changes vs V2
- Depth scaled from 7 → 12 transformer layers
- Internal FFN `dim_feedforward` nearly doubled: 1648 → 3072 (≈ 4× embedding dim, the "standard" GPT ratio)
- Reverted to Shakespeare corpus

### Performance (as of ~16 epochs)
- **Loss:** ~4.5 (plateauing)
- **Behavior:** Sentences are grammatically coherent and stylistically Shakespearean in structure. However the model lacks deeper contextual understanding — word choices within a sentence are often locally plausible but semantically disconnected from the broader context. The model appears to have learned word-level patterns and common phrase templates without capturing meaning or discourse-level coherence.

### Notes on the higher loss vs V2
The ~4.5 loss vs V2's ~1.5 is expected and not necessarily a regression:
- Shakespeare has a substantially larger, more archaic vocabulary (~24,943 tokens vs ~15,000 for the Kafka corpus)
- Higher theoretical entropy in the target distribution → higher irreducible loss floor
- The plateau may indicate the model has saturated its representational capacity at the current embedding dimension (768) relative to the vocabulary size and corpus complexity
