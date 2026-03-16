"""
diagnostics.py — ShakespeareLM model diagnostic tool

Usage:
    python utils/diagnostics.py versions/v2_kafka_12l/checkpoints/transformer_dev_e3_b500.pt
    python utils/diagnostics.py versions/v2_kafka_12l/checkpoints/transformer_dev_e3_b500.pt --batches 200 --batch-size 16

Runs validation data through a checkpoint and prints extensive diagnostics
intended to be fed to an LLM for architectural analysis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import io
import math
from collections import Counter, defaultdict

# Force UTF-8 output so box-drawing chars and dashes don't crash on Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import importlib.util

from dataset import ShakespeareDataset, collate_fn
from tokenizer import Tokenizer
from config import get_config
from torch.utils.data import DataLoader

cfg = get_config()


def load_model_class(model_file):
    spec = importlib.util.spec_from_file_location("model", model_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ShakespeareLM


ShakespeareLM = load_model_class(cfg["model_file"])

SEP = "=" * 80


def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def sub(title):
    print(f"\n  --- {title} ---")


# ─────────────────────────────────────────────────────────────────────────────
# Architecture detection from state dict
# ─────────────────────────────────────────────────────────────────────────────

def detect_arch(sd):
    vocab_size = sd["embedding_layer.weight"].shape[0]
    emb_dim    = sd["embedding_layer.weight"].shape[1]
    num_layers = 0
    while f"transformer_encoder.layers.{num_layers}.self_attn.in_proj_weight" in sd:
        num_layers += 1
    ffn_key     = "transformer_encoder.layers.0.linear1.weight"
    hidden_size = sd[ffn_key].shape[0] if ffn_key in sd else 1648
    return vocab_size, emb_dim, num_layers, hidden_size


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ShakespeareLM diagnostic tool")
    parser.add_argument("checkpoint", help="Path to .pt checkpoint file")
    parser.add_argument("--batches",    type=int, default=100,
                        help="Validation batches to run (default: 200)")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: '{args.checkpoint}' not found")
        sys.exit(1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ── Load ─────────────────────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer      = Tokenizer()
    vocab_size_tok = len(tokenizer.tokens)

    print(f"Loading checkpoint: {args.checkpoint}")
    sd = torch.load(args.checkpoint, map_location=device, weights_only=True)

    vocab_size, emb_dim, num_layers, hidden_size = detect_arch(sd)
    model = ShakespeareLM(
        vocab_size=vocab_size,
        embedding_dim=emb_dim,
        num_layers=num_layers,
        hidden_size=hidden_size,
    ).to(device)
    model.load_state_dict(sd)
    model.eval()

    print("Setting up validation dataset...")
    val_ds   = ShakespeareDataset(tokenizer=tokenizer, split="test")
    val_iter = iter(DataLoader(val_ds, batch_size=args.batch_size,
                               shuffle=True, collate_fn=collate_fn))

    crit_none = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    crit_mean = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")

    # ── Section 1: Architecture ───────────────────────────────────────────────
    section("1. MODEL ARCHITECTURE")
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emb_params       = sum(p.numel() for p in model.embedding_layer.parameters())
    tfm_params       = sum(p.numel() for p in model.transformer_encoder.parameters())
    head_params      = sum(p.numel() for p in model.output_head.parameters())
    ffn_ratio        = hidden_size / emb_dim

    print(f"  Checkpoint file:           {args.checkpoint}")
    print(f"  Vocab size (model):        {vocab_size:,}")
    print(f"  Vocab size (tokenizer):    {vocab_size_tok:,}")
    print(f"  Vocab mismatch:            {'YES — may cause OOB errors' if vocab_size != vocab_size_tok else 'no'}")
    print(f"  Embedding dim:             {emb_dim}")
    print(f"  Transformer layers:        {num_layers}")
    print(f"  FFN hidden size:           {hidden_size}  (ratio = {ffn_ratio:.2f}x emb_dim; standard GPT-2 = 4.0x)")
    print(f"  Total parameters:          {total_params:,}")
    print(f"  Trainable parameters:      {trainable_params:,}")
    print(f"  Embedding params:          {emb_params:,}  ({emb_params/total_params*100:.1f}%)")
    print(f"  Transformer body params:   {tfm_params:,}  ({tfm_params/total_params*100:.1f}%)")
    print(f"  Output head params:        {head_params:,}  ({head_params/total_params*100:.1f}%)")
    print(f"  Head/body param ratio:     {head_params/max(tfm_params,1):.2f}x  (>1 means head dominates — high bias risk)")

    sub("Per-leaf-module parameter counts")
    print(f"  {'Module':<58} {'Params':>12}")
    for name, mod in model.named_modules():
        if not list(mod.children()):
            p = sum(x.numel() for x in mod.parameters())
            if p > 0:
                print(f"  {name:<58} {p:>12,}")

    # ── Section 2: Weight statistics ──────────────────────────────────────────
    section("2. WEIGHT STATISTICS")
    print(f"  {'Parameter':<55} {'mean':>9} {'std':>9} {'min':>9} {'max':>9} {'L2norm':>10} {'%|w|<1e-4':>10}")
    print("  " + "-" * 115)
    for name, param in model.named_parameters():
        d = param.detach().float()
        print(f"  {name:<55} "
              f"{d.mean().item():>9.5f} "
              f"{d.std().item():>9.5f} "
              f"{d.min().item():>9.5f} "
              f"{d.max().item():>9.5f} "
              f"{d.norm().item():>10.3f} "
              f"{(d.abs() < 1e-4).float().mean().item()*100:>9.2f}%")

    # ── Register activation hooks ─────────────────────────────────────────────
    act_stats = {}

    def make_hook(lname):
        def hook(mod, inp, out):
            if isinstance(out, torch.Tensor):
                o = out.detach().float()
                act_stats[lname] = {
                    "mean":    o.mean().item(),
                    "std":     o.std().item(),
                    "max_abs": o.abs().max().item(),
                    "pct0":    (o.abs() < 1e-3).float().mean().item() * 100,
                }
        return hook

    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, nn.LayerNorm, nn.GELU)):
            hooks.append(mod.register_forward_hook(make_hook(name)))

    # ── Section 3: Collect validation metrics ─────────────────────────────────
    section("3. RUNNING VALIDATION  (this may take a moment...)")
    print(f"  batches={args.batches}, batch_size={args.batch_size}")

    batch_losses, token_losses_all                = [], []
    top1_correct_all, top5_correct_all            = [], []
    ranks_all, max_probs_all, entropies_all        = [], [], []
    pred_tokens_all, true_tokens_all               = [], []
    pos_losses                                     = defaultdict(list)
    logit_maxes                                    = []
    sampled_dists                                  = []   # for cosine-sim analysis

    with torch.no_grad():
        for bi in range(args.batches):
            if bi > 0 and bi % 20 == 0:
                print(f"  [{bi}/{args.batches} batches]  tokens so far: {len(token_losses_all):,}")
            try:
                inp, tgt = next(val_iter)
            except StopIteration:
                val_iter = iter(DataLoader(val_ds, batch_size=args.batch_size,
                                           shuffle=True, collate_fn=collate_fn))
                inp, tgt = next(val_iter)

            inp, tgt = inp.to(device), tgt.to(device)
            logits   = model(inp)          # (B, T, V)
            B, T, V  = logits.shape

            # Batch loss
            batch_losses.append(crit_mean(logits.reshape(-1, V), tgt.reshape(-1)).item())

            # Per-token losses (vectorised)
            tok_loss  = crit_none(logits.reshape(-1, V), tgt.reshape(-1))
            mask_flat = tgt.reshape(-1) != -100
            tok_loss_m = tok_loss[mask_flat]
            token_losses_all.extend(tok_loss_m.tolist())

            # Position-level losses (vectorised)
            positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T).reshape(-1)
            pos_m     = positions[mask_flat]
            for pos, lv in zip(pos_m.tolist(), tok_loss_m.tolist()):
                pos_losses[pos].append(lv)

            # Probabilities on valid tokens
            logits_m = logits.reshape(-1, V)[mask_flat]
            tgt_m    = tgt.reshape(-1)[mask_flat]
            probs    = F.softmax(logits_m, dim=-1)
            log_prob = F.log_softmax(logits_m, dim=-1)

            max_probs_all.extend(probs.max(dim=-1).values.tolist())
            logit_maxes.append(logits_m.abs().max().item())

            entropy = -(probs * log_prob).sum(dim=-1)
            entropies_all.extend(entropy.tolist())

            # Top-1
            top1 = logits_m.argmax(dim=-1)
            top1_correct_all.extend((top1 == tgt_m).float().tolist())
            pred_tokens_all.extend(top1.tolist())
            true_tokens_all.extend(tgt_m.tolist())

            # Top-5
            top5 = logits_m.topk(5, dim=-1).indices
            top5_correct_all.extend((top5 == tgt_m.unsqueeze(-1)).any(-1).float().tolist())

            # Rank of target token (vectorised: rank = #tokens with higher logit + 1)
            target_logits = logits_m.gather(1, tgt_m.unsqueeze(1))   # (N, 1)
            ranks_batch   = (logits_m > target_logits).sum(dim=1) + 1  # (N,)
            ranks_all.extend(ranks_batch.tolist())

            # Save a sample distribution for cosine-sim analysis
            if bi < 50:
                sampled_dists.append(probs[-1].cpu())

    for h in hooks:
        h.remove()

    # ── Section 4: Core metrics ───────────────────────────────────────────────
    section("4. CORE VALIDATION METRICS")
    mean_loss   = float(np.mean(batch_losses))
    std_loss    = float(np.std(batch_losses))
    ppl         = math.exp(min(mean_loss, 20))
    top1_acc    = float(np.mean(top1_correct_all)) * 100
    top5_acc    = float(np.mean(top5_correct_all)) * 100
    mean_rank   = float(np.mean(ranks_all))
    median_rank = float(np.median(ranks_all))
    mrr         = float(np.mean([1.0 / r for r in ranks_all]))
    random_loss = math.log(vocab_size)

    print(f"  Batches evaluated:            {args.batches}")
    print(f"  Total tokens evaluated:       {len(token_losses_all):,}")
    print(f"  Random-baseline loss:         {random_loss:.4f}  (= log vocab_size)")
    print(f"  Mean batch loss:              {mean_loss:.4f}")
    print(f"  Std of batch losses:          {std_loss:.4f}")
    print(f"  Perplexity:                   {ppl:.2f}  (random baseline = {math.exp(random_loss):.0f})")
    print(f"  Top-1 accuracy:               {top1_acc:.2f}%")
    print(f"  Top-5 accuracy:               {top5_acc:.2f}%")
    print(f"  Mean rank of target token:    {mean_rank:.1f}  (random baseline = {vocab_size//2:,})")
    print(f"  Median rank of target token:  {median_rank:.1f}")
    print(f"  Mean reciprocal rank (MRR):   {mrr:.5f}  (max = 1.0)")

    sub("Token-level loss distribution (percentiles)")
    la = np.array(token_losses_all)
    for pct in [10, 25, 50, 75, 90, 95, 99]:
        print(f"    p{pct:<3}: {np.percentile(la, pct):.4f}")
    print(f"    max:  {la.max():.4f}")

    # ── Section 5: Bias / Variance / Collapse ─────────────────────────────────
    section("5. BIAS / VARIANCE / COLLAPSE ANALYSIS")
    pred_counter   = Counter(pred_tokens_all)
    true_counter   = Counter(true_tokens_all)
    unique_preds   = len(pred_counter)
    n_total        = len(pred_tokens_all)
    top10_preds    = pred_counter.most_common(10)
    top10_coverage = sum(c for _, c in top10_preds) / n_total * 100
    top20_pred_set = {t for t, _ in pred_counter.most_common(20)}
    top20_true_set = {t for t, _ in true_counter.most_common(20)}
    overlap        = top20_pred_set & top20_true_set

    print(f"  Unique tokens predicted (top-1):         {unique_preds:,}  of {vocab_size:,} vocab  "
          f"({unique_preds/vocab_size*100:.1f}%)")
    print(f"  Top-10 predicted tokens cover:           {top10_coverage:.1f}% of all predictions")
    print(f"  Top-20 predicted intersect top-20 true tokens:  {len(overlap)}/20  "
          f"(high = predicting by frequency not context)")

    sub("Top-20 most predicted tokens (top-1)")
    print(f"  {'Rank':<5} {'ID':<8} {'Word':<22} {'Count':<10} {'%':>6}")
    for rank, (tid, cnt) in enumerate(pred_counter.most_common(20), 1):
        word = tokenizer.untokenize_text([tid])[0]
        print(f"  {rank:<5} {tid:<8} {word:<22} {cnt:<10} {cnt/n_total*100:>6.2f}%")

    sub("Top-20 most frequent TRUE tokens in validation data")
    print(f"  {'Rank':<5} {'ID':<8} {'Word':<22} {'Count':<10} {'%':>6}")
    for rank, (tid, cnt) in enumerate(true_counter.most_common(20), 1):
        word = tokenizer.untokenize_text([tid])[0]
        print(f"  {rank:<5} {tid:<8} {word:<22} {cnt:<10} {cnt/n_total*100:>6.2f}%")

    # ── Section 6: Entropy ────────────────────────────────────────────────────
    section("6. PREDICTED DISTRIBUTION ENTROPY")
    ea             = np.array(entropies_all)
    max_entropy    = math.log(vocab_size)
    collapse_ratio = (ea < 1.0).mean() * 100

    print(f"  Max possible entropy (uniform):    {max_entropy:.4f}  (= log vocab_size)")
    print(f"  Mean entropy:                      {ea.mean():.4f}  ({ea.mean()/max_entropy*100:.1f}% of max)")
    print(f"  Std of entropy across tokens:      {ea.std():.4f}  (low = same confidence for all inputs)")
    print(f"  Median entropy:                    {np.median(ea):.4f}")
    print(f"  Min entropy:                       {ea.min():.4f}")
    print(f"  Max entropy:                       {ea.max():.4f}")
    print(f"  % tokens with entropy < 1.0:       {collapse_ratio:.1f}%  (collapse indicator)")
    print()
    print(f"  INTERPRETATION:")
    print(f"    entropy near 0          = model hyper-confident on one token (collapsed)")
    print(f"    entropy near {max_entropy:.1f}   = model is near-uniform (untrained)")
    print(f"    low entropy std         = model outputs same distribution for all inputs (high bias)")

    # ── Section 7: Max softmax probability ───────────────────────────────────
    section("7. MAX SOFTMAX PROBABILITY (CONFIDENCE)")
    mpa = np.array(max_probs_all)
    print(f"  Random baseline:          {1/vocab_size:.6f}")
    print(f"  Mean max probability:     {mpa.mean():.4f}")
    print(f"  Std max probability:      {mpa.std():.4f}")
    print(f"  Median max probability:   {np.median(mpa):.4f}")
    print(f"  % with max_prob > 0.1:    {(mpa > 0.1).mean()*100:.1f}%")
    print(f"  % with max_prob > 0.5:    {(mpa > 0.5).mean()*100:.1f}%")
    print(f"  % with max_prob > 0.9:    {(mpa > 0.9).mean()*100:.1f}%")

    # ── Section 8: Distribution cosine similarity ─────────────────────────────
    section("8. DISTRIBUTION COSINE SIMILARITY  (collapse indicator)")
    mean_sim = None
    std_sim  = None
    if len(sampled_dists) >= 10:
        dists      = torch.stack(sampled_dists[:50])
        dists_norm = F.normalize(dists, dim=-1)
        n          = len(dists_norm)
        sim_mat    = dists_norm @ dists_norm.T
        upper_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        pair_sims  = sim_mat[upper_mask]
        mean_sim   = pair_sims.mean().item()
        std_sim    = pair_sims.std().item()
        print(f"  Mean pairwise cosine similarity:  {mean_sim:.4f}")
        print(f"  Std pairwise cosine similarity:   {std_sim:.4f}")
        print(f"  Min pairwise cosine similarity:   {pair_sims.min().item():.4f}")
        print(f"  Max pairwise cosine similarity:   {pair_sims.max().item():.4f}")
        print()
        print(f"  INTERPRETATION:")
        print(f"    ~1.0 = all inputs produce identical distributions (fully collapsed / high bias)")
        print(f"    ~0.0 = distributions are orthogonal (diverse, context-sensitive)")
    else:
        print("  Not enough samples collected.")

    # ── Section 9: Calibration ────────────────────────────────────────────────
    section("9. CALIBRATION  (confidence vs. accuracy)")
    mpa_all  = np.array(max_probs_all)
    top1_all = np.array(top1_correct_all)
    ece      = 0.0
    print(f"  {'Confidence bucket':<22} {'Mean conf':>10} {'Accuracy':>10} {'|conf-acc|':>12} {'N':>8}")
    for lo in np.arange(0.0, 1.0, 0.1):
        hi   = round(lo + 0.1, 2)
        mask = (mpa_all >= lo) & (mpa_all < hi)
        if mask.sum() > 0:
            conf = mpa_all[mask].mean()
            acc  = top1_all[mask].mean()
            err  = abs(conf - acc)
            ece += mask.mean() * err
            print(f"  [{lo:.1f} – {hi:.1f}]             "
                  f"{conf:>10.4f} {acc*100:>9.1f}%  {err:>12.4f}  {mask.sum():>8,}")
    print(f"\n  Expected Calibration Error (ECE): {ece:.4f}  (<0.05 good, >0.15 miscalibrated)")

    # ── Section 10: KL divergence predicted vs. true token distribution ───────
    section("10. PREDICTED vs. TRUE TOKEN DISTRIBUTION  (frequency bias)")
    pred_counts = np.zeros(vocab_size)
    true_counts = np.zeros(vocab_size)
    for tid, cnt in pred_counter.items():
        if tid < vocab_size:
            pred_counts[tid] = cnt
    for tid, cnt in true_counter.items():
        if tid < vocab_size:
            true_counts[tid] = cnt

    eps          = 1e-10
    pred_dist    = pred_counts / (pred_counts.sum() + eps)
    true_dist    = true_counts / (true_counts.sum() + eps)
    kl_pred_true = float(np.sum(true_dist * (np.log(true_dist + eps) - np.log(pred_dist + eps))))
    over_pred    = int((pred_dist > true_dist * 1.5).sum())
    under_pred   = int((pred_dist < true_dist * 0.5).sum())
    vocab_used   = int((pred_counts > 0).sum())

    print(f"  KL(true || predicted):              {kl_pred_true:.4f}  (~0 = distributions match)")
    print(f"  Tokens over-predicted  (>1.5x):     {over_pred:,}")
    print(f"  Tokens under-predicted (<0.5x):     {under_pred:,}")
    print(f"  Tokens never predicted:             {vocab_size - vocab_used:,}")
    print(f"  Effective vocabulary used:          {vocab_used:,}  of {vocab_size:,}  "
          f"({vocab_used/vocab_size*100:.1f}%)")

    # ── Section 11: Loss by sequence position ─────────────────────────────────
    section("11. LOSS BY SEQUENCE POSITION")
    print(f"  {'Positions':<14} {'Mean loss':>10} {'Tokens':>10}  note")
    for lo, hi in [(0, 5), (5, 15), (15, 30), (30, 50), (50, 75), (75, 101)]:
        bucket = []
        for pos in range(lo, hi):
            bucket.extend(pos_losses.get(pos, []))
        if bucket:
            bml  = np.mean(bucket)
            note = " <-- higher than average" if bml > mean_loss * 1.2 else ""
            print(f"  pos {lo:02d}–{hi:02d}       {bml:>10.4f} {len(bucket):>10,}  {note}")

    # ── Section 12: Logit statistics ──────────────────────────────────────────
    section("12. LOGIT STATISTICS  (numerical stability)")
    print(f"  Mean max |logit| across batches:  {np.mean(logit_maxes):.4f}")
    print(f"  Max  max |logit| across batches:  {np.max(logit_maxes):.4f}")
    print(f"  (>50 suggests logit explosion — softmax saturates to a single token)")

    # ── Section 13: Activation statistics ────────────────────────────────────
    section("13. ACTIVATION STATISTICS  (last batch, from forward hooks)")
    if act_stats:
        print(f"  {'Layer':<58} {'mean':>9} {'std':>9} {'max|a|':>9} {'%~0':>7}")
        print("  " + "-" * 100)
        for lname, st in act_stats.items():
            print(f"  {lname:<58} {st['mean']:>9.5f} {st['std']:>9.5f} "
                  f"{st['max_abs']:>9.4f} {st['pct0']:>6.2f}%")

    # ── Section 14: Gradient health ───────────────────────────────────────────
    section("14. GRADIENT HEALTH  (single backward pass on one validation batch)")
    model.train()
    try:
        g_iter     = iter(DataLoader(val_ds, batch_size=args.batch_size,
                                     shuffle=True, collate_fn=collate_fn))
        inp_g, tgt_g = next(g_iter)
        inp_g, tgt_g = inp_g.to(device), tgt_g.to(device)
        model.zero_grad()
        logits_g = model(inp_g)
        loss_g   = crit_mean(logits_g.reshape(-1, vocab_size), tgt_g.reshape(-1))
        loss_g.backward()

        total_norm = 0.0
        print(f"  {'Parameter':<55} {'grad_norm':>12} {'weight_norm':>12} {'ratio':>10}")
        print("  " + "-" * 95)
        for name, param in model.named_parameters():
            if param.grad is not None:
                gn         = param.grad.norm().item()
                wn         = param.data.norm().item()
                total_norm += gn ** 2
                ratio      = gn / (wn + 1e-10)
                flag       = ""
                if gn < 1e-6:
                    flag = "  <-- VANISHING"
                elif gn > 10:
                    flag = "  <-- EXPLODING"
                print(f"  {name:<55} {gn:>12.6f} {wn:>12.4f} {ratio:>10.6f}{flag}")

        total_norm = math.sqrt(total_norm)
        print(f"\n  Total gradient norm (pre-clip): {total_norm:.6f}")
        print(f"  (Your grad_clip = 0.5; norm >> 0.5 means clipping fires every step)")
    except Exception as e:
        print(f"  Could not run gradient pass: {e}")
    model.eval()

    # ── Section 15: Summary & diagnostic flags ────────────────────────────────
    section("15. SUMMARY & DIAGNOSTIC FLAGS")

    flags = []

    if mean_loss > random_loss * 0.95:
        flags.append(f"CRITICAL      Loss {mean_loss:.3f} is near random baseline {random_loss:.3f} — model has learned almost nothing")
    elif mean_loss > random_loss * 0.7:
        flags.append(f"WARNING       Loss {mean_loss:.3f} is still very close to random — minimal learning")

    if ppl > 500:
        flags.append(f"CRITICAL      Perplexity {ppl:.0f} — near-random next-token prediction")

    if top1_acc < 5.0:
        flags.append(f"WARNING       Top-1 accuracy {top1_acc:.2f}% is very low")

    if unique_preds < 100:
        flags.append(f"CRITICAL      Only {unique_preds} unique top-1 predictions — severe mode collapse")
    elif unique_preds < 500:
        flags.append(f"WARNING       Only {unique_preds} unique top-1 predictions — low diversity / high bias")

    if top10_coverage > 80:
        flags.append(f"WARNING       Top-10 predicted tokens cover {top10_coverage:.1f}% of predictions — very low diversity")

    if ea.mean() < 1.0:
        flags.append(f"CRITICAL      Mean entropy {ea.mean():.3f} < 1.0 — model is over-confident / collapsed")
    elif ea.mean() < 2.0:
        flags.append(f"WARNING       Mean entropy {ea.mean():.3f} is low — possible over-confidence")

    if ea.std() < 0.3:
        flags.append(f"WARNING       Entropy std {ea.std():.3f} is very low — near-identical distributions for all inputs (high bias)")

    if collapse_ratio > 20:
        flags.append(f"WARNING       {collapse_ratio:.1f}% of predictions have entropy < 1.0 — frequent mode collapse")

    if mean_sim is not None:
        if mean_sim > 0.95:
            flags.append(f"CRITICAL      Mean cosine similarity {mean_sim:.3f} — outputs near-identical regardless of input (collapsed)")
        elif mean_sim > 0.80:
            flags.append(f"WARNING       Mean cosine similarity {mean_sim:.3f} — low context sensitivity (high bias)")

    if len(overlap) >= 15:
        flags.append(f"WARNING       Top-20 predicted intersect top-20 true = {len(overlap)}/20 — predicting by frequency, not context")

    if kl_pred_true > 2.0:
        flags.append(f"WARNING       KL(true||pred) = {kl_pred_true:.3f} — predicted distribution badly mismatches ground truth")

    if vocab_used / vocab_size < 0.30:
        flags.append(f"WARNING       Only {vocab_used/vocab_size*100:.1f}% of vocabulary ever predicted — model ignoring most tokens")

    if ece > 0.15:
        flags.append(f"WARNING       ECE {ece:.3f} > 0.15 — model is miscalibrated (confidence ≠ accuracy)")

    if num_layers < 4:
        flags.append(f"ARCHITECTURE  Only {num_layers} transformer layers — very shallow; limits context-sensitive prediction")
    if ffn_ratio < 3.0:
        flags.append(f"ARCHITECTURE  FFN ratio {ffn_ratio:.2f}x < standard 4.0x — reduced per-layer capacity")
    if head_params > tfm_params * 2:
        flags.append(f"ARCHITECTURE  Output head has {head_params/tfm_params:.1f}x more params than transformer body — head likely dominates learning")

    if flags:
        for f in flags:
            tag   = f.split()[0]
            badge = {"CRITICAL": "!!!", "WARNING": " ! ", "ARCHITECTURE": "ARC"}.get(tag, "   ")
            print(f"  [{badge}] {f}")
    else:
        print("  No flags raised.")

    print()
    print("  ── Quick reference ──────────────────────────────────────────────────────")
    print(f"  Loss={mean_loss:.3f} (rand={random_loss:.2f})  PPL={ppl:.1f}  "
          f"Top1={top1_acc:.1f}%  Top5={top5_acc:.1f}%")
    print(f"  MRR={mrr:.4f}  Rank={mean_rank:.0f}/{vocab_size}  "
          f"ECE={ece:.3f}  KL={kl_pred_true:.3f}")
    print(f"  Entropy: mean={ea.mean():.3f} std={ea.std():.3f} collapse={collapse_ratio:.1f}%")
    print(f"  UniquePreds={unique_preds}/{vocab_size} ({unique_preds/vocab_size*100:.1f}%)  "
          f"VocabUsed={vocab_used/vocab_size*100:.1f}%")
    if mean_sim is not None:
        print(f"  DistCosineSim: mean={mean_sim:.3f} std={std_sim:.3f}")
    print(f"  Layers={num_layers}  FFNratio={ffn_ratio:.2f}x  "
          f"HeadParams={head_params:,} ({head_params/total_params*100:.1f}%)  "
          f"BodyParams={tfm_params:,} ({tfm_params/total_params*100:.1f}%)")
    print()


if __name__ == "__main__":
    main()
