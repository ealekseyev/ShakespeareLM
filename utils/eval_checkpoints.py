"""
Evaluate all checkpoints in the active version's checkpoint dir on 64 validation batches each.
Scoring: rank-decay score (primary), top-1 accuracy, top-5 accuracy.
  - Correct token at rank 1 → 1.0
  - Correct token at rank 2 → 0.8
  - Correct token at rank 3 → 0.6
  - Correct token at rank 4 → 0.4
  - Correct token at rank 5 → 0.2
  - Not in top 5         → 0.0
Padding positions (target == -100) are skipped.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import re
import queue
import threading
import importlib.util
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import get_config
from tokenizer import Tokenizer
from dataset import ShakespeareDataset, collate_fn

cfg = get_config()


def load_model_class(model_file):
    spec = importlib.util.spec_from_file_location("model", model_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ShakespeareLM


ShakespeareLM = load_model_class(cfg["model_file"])

CHECKPOINT_DIR = cfg["checkpoint_dir"]
NUM_EVAL_BATCHES = 64
BATCH_SIZE = 32

device = torch.device("xpu")
print(f"Using {device}\n")

tokenizer = Tokenizer()
VOCAB_SIZE = len(tokenizer.inv_tokens) + 1
PAD_TOKEN = len(tokenizer.inv_tokens)

# Build a fixed validation set (same batches for every model)
print("Building validation dataloader...")
val_dataset = ShakespeareDataset(tokenizer=tokenizer, split="test", sequences_per_epoch=NUM_EVAL_BATCHES * BATCH_SIZE)
_collate = lambda b: collate_fn(b, pad_token_id=PAD_TOKEN)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate)
val_batches = []
for i, batch in enumerate(val_loader):
    val_batches.append(batch)
    if i + 1 >= NUM_EVAL_BATCHES:
        break
print(f"Cached {len(val_batches)} validation batches.\n")

RANK_SCORES = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2}

def evaluate(model):
    model.eval()
    total_score = 0.0
    top1_correct = 0
    top5_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for input_ids, target_ids in val_batches:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            logits = model(input_ids)  # (B, T, V)

            # ranked indices descending by logit, shape (B, T, V)
            ranked = torch.argsort(logits, dim=-1, descending=True)

            flat_targets = target_ids.reshape(-1)        # (B*T,)
            flat_ranked = ranked.reshape(-1, ranked.size(-1))  # (B*T, V)

            valid = flat_targets != -100
            flat_targets = flat_targets[valid]
            flat_ranked = flat_ranked[valid]

            # rank of correct token at each position (1-indexed)
            # flat_ranked[i] is the sorted vocab for position i
            # we need where flat_targets[i] appears in flat_ranked[i]
            correct = (flat_ranked == flat_targets.unsqueeze(1))  # (N, V)
            ranks = correct.int().argmax(dim=1) + 1              # (N,) 1-indexed

            for r in ranks.tolist():
                total_score += RANK_SCORES.get(r, 0.0)
                if r == 1:
                    top1_correct += 1
                if r <= 5:
                    top5_correct += 1

            total_tokens += len(flat_targets)

    rank_score = total_score / total_tokens
    top1 = top1_correct / total_tokens
    top5 = top5_correct / total_tokens
    return rank_score, top1, top5


checkpoint_pattern = re.compile(r"transformer_dev_e(\d+)_b(\d+)\.pt")

checkpoints = []
for fname in sorted(os.listdir(CHECKPOINT_DIR)):
    m = checkpoint_pattern.match(fname)
    if m and int(m.group(2)) == 0:
        checkpoints.append((int(m.group(1)), int(m.group(2)), os.path.join(CHECKPOINT_DIR, fname)))
checkpoints.sort()
print(f"Found {len(checkpoints)} checkpoints to evaluate.\n")

# Background thread loads state dicts from disk to CPU while main thread evaluates on XPU.
# maxsize=2 keeps one model preloaded and ready at all times without reading too far ahead.
prefetch_queue = queue.Queue(maxsize=2)

def loader_fn():
    for epoch, batch_num, path in checkpoints:
        try:
            state = torch.load(path, map_location="cpu")
            prefetch_queue.put((epoch, path, state, None))
        except Exception as e:
            prefetch_queue.put((epoch, path, None, str(e)))
    prefetch_queue.put(None)  # sentinel

threading.Thread(target=loader_fn, daemon=True).start()

results = []

while True:
    item = prefetch_queue.get()
    if item is None:
        break
    epoch, path, state, err = item
    if err:
        print(f"  SKIP {path}: {err}")
        continue
    try:
        model = ShakespeareLM(vocab_size=VOCAB_SIZE, pad_token_id=PAD_TOKEN)
        model.load_state_dict(state, strict=True)
        model.to(device)

        rank_score, top1, top5 = evaluate(model)
        results.append((rank_score, top1, top5, epoch, path))
        print(f"  e{epoch:>2}_b0   rank_score={rank_score:.4f}  top1={top1:.3f}  top5={top5:.3f}")

    except Exception as e:
        print(f"  SKIP {path}: {e}")

results.sort(key=lambda x: x[0], reverse=True)  # higher rank_score is better

print("\n" + "=" * 60)
print("TOP 10 CHECKPOINTS BY RANK-DECAY SCORE")
print("=" * 60)
for i, (rank_score, top1, top5, epoch, path) in enumerate(results[:10], 1):
    print(f"  #{i:>2}  e{epoch:>2}_b0   rank_score={rank_score:.4f}  top1={top1:.3f}  top5={top5:.3f}  ({path})")
