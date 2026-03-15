from torch.utils.data import DataLoader
from dataset import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model_transformer_revised import *
from tokenizer import *
from datetime import datetime
import os
import re

# improved hyperparameters
initial_lr = 3e-3
e10_lr = 1e-4
warmup_steps = 1000  # Linear warmup

dropout = 0.05 # Keep at 0.05 for better learning
batch_size = 64   # Reduced for better gradient estimates
max_grad_norm = 1.0  # Gradient clipping

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")
print("Using", device)

tokenizer = Tokenizer()
VOCAB_SIZE = len(tokenizer.inv_tokens) + 1  # +1 for padding token slot
PAD_TOKEN = len(tokenizer.inv_tokens)       # padding ID = first slot beyond real vocab

train_dataset = ShakespeareDataset(tokenizer=tokenizer)
test_dataset = ShakespeareDataset(tokenizer=tokenizer)

_collate = lambda b: collate_fn(b, pad_token_id=PAD_TOKEN)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate)
test_dataloader_iter = iter(DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=_collate))

RESUME = True  # Set to False to start from scratch

def find_latest_checkpoint(folder="checkpoints"):
    best = None
    pattern = re.compile(r"transformer_dev_e(\d+)_b(\d+)\.pt")
    for fname in os.listdir(folder):
        m = pattern.match(fname)
        if m:
            e, b = int(m.group(1)), int(m.group(2))
            if best is None or (e, b) > best[:2]:
                best = (e, b, os.path.join(folder, fname))
    return best  # (epoch, batch, path) or None

model = ShakespeareLM(dropout=dropout, vocab_size=VOCAB_SIZE, pad_token_id=PAD_TOKEN).to(device)
start_epoch = 0
if RESUME:
    latest = find_latest_checkpoint()
    if latest:
        e, b, path = latest
        model.load_state_dict(torch.load(path, map_location=device))
        start_epoch = e
        print(f"Resumed from {path} (epoch {e}, batch {b})")
    else:
        print("No checkpoint found, starting from scratch")
else:
    print("Model initialized")
criterion = nn.CrossEntropyLoss(label_smoothing=0.05, ignore_index=-100)  # Reduced label smoothing, ignore padding
optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=0.0001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50 * 1000, eta_min=1e-6)  # Cosine annealing

# Warmup scheduler
warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
step_count = 0

scaler = torch.amp.GradScaler(device.type)

def adam_update_lr(o, new_lr):
    for param_group in o.param_groups:
        param_group['lr'] = new_lr  # New learning rate

def rank_tensor_indices(tensor):
    # Flatten the tensor in case it's not 1D
    tensor = tensor.flatten()
    # Sort the tensor in descending order and return the indices
    _, sorted_indices = torch.sort(tensor, descending=True)
    return sorted_indices.tolist()

NUM_EPOCHS = 50
for i in range(start_epoch, NUM_EPOCHS):
    if i == 10:
        adam_update_lr(optimizer, e10_lr)
    # train
    model.train()
    try:
        batch = 0
        for input_ids, target_ids in train_dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(input_ids)  # (batch_size, seq_len, vocab_size)
                loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

            scaler.scale(loss).backward()

            # Unscale before grad clipping so clip threshold is in real gradient units
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            
            # Learning rate scheduling
            step_count += 1
            if step_count <= warmup_steps:
                warmup_scheduler.step()
            else:
                scheduler.step()
            if batch == 0:
                model.eval()
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16):
                    input_ids, target_ids = next(test_dataloader_iter)
                    input_ids = input_ids.to(device)
                    target_ids = target_ids.to(device)
                    logits = model(input_ids)  # (batch_size, seq_len, vocab_size)

                    # Validation loss
                    val_loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

                    # Get predictions for the last position
                    last_logits = logits[0, -1, :]  # Last position logits for first sample
                    prob_distribution = F.softmax(last_logits.cpu().unsqueeze(0), dim=-1)
                    batch_sampled = top_p_sample_batch(prob_distribution)

                    # Get input and target words for display
                    input_words = tokenizer.untokenize_text(input_ids[0].cpu().numpy())
                    target_word = tokenizer.untokenize_text(target_ids[0, -1:].cpu().numpy())  # Last target word

                    preds = tokenizer.untokenize_text(rank_tensor_indices(prob_distribution[0])[0:5])
                    top_p_res = tokenizer.untokenize_text([batch_sampled[0]])

                    current_lr = optimizer.param_groups[0]['lr']
                    t = datetime.now().strftime("%H:%M:%S")
                    print(f"[{t}] e{i}b{batch}, TrainLoss {round(float(loss.detach()), 2)}, ValLoss {round(float(val_loss), 2)}, LR {current_lr:.2e} | Target: {target_word[0] if target_word else 'N/A'} | Pred: {top_p_res[0]} - [{', '.join(preds)}] | Context: {' '.join(input_words[-10:])}")
                model.train()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elif device.type == "xpu":
                    torch.xpu.synchronize()
                torch.save(model.state_dict(), f"checkpoints/transformer_dev_e{i}_b{batch}.pt")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] e{i}b{batch} loss: {round(float(loss.detach()), 2)}")
            batch += 1
    except Exception as e:
        print(e)
        breakpoint()
    # # test model
    # model.eval()
    # test_loss = 0.0
    # correct_preds = 0
    # total_preds = 0
    # with torch.no_grad():
    #     input_ids, attention_mask, labels = next(iter(test_dataloader))
    #     preds = model(input_ids.to(device), attention_mask.to(device))
    #     test_loss = criterion(preds, labels.to(device))
    #     # compute accuracy
    #     preds = (preds.cpu() > 0.5).float()
    #     correct_preds += (preds.cpu() == labels.cpu()).sum().item()
    #     total_preds += labels.cpu().size(0)
    # print(f"Epoch {i} test loss: {test_loss.item()} | Accuracy: {correct_preds / total_preds:.3f}")
