import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np

from dataset import PIIDataset, collate_batch
from labels import LABELS, LABEL2ID
from model import create_model
from collections import Counter


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--combine_train_dev", action="store_true", 
                    help="Combine train and dev sets for training (use for very small datasets)")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--patience", type=int, default=20, help="Early stopping patience (set high for small datasets)")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    return ap.parse_args()


def evaluate(model, dev_dl, device, class_weights=None):
    """Evaluate model on dev set and return average loss"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dev_dl:
            input_ids = torch.tensor(batch["input_ids"], device=device)
            attention_mask = torch.tensor(batch["attention_mask"], device=device)
            labels = torch.tensor(batch["labels"], device=device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            if class_weights is not None:
                # Use weighted loss for consistency
                logits = outputs.logits
                loss_fct = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
                loss = loss_fct(logits.view(-1, len(LABELS)), labels.view(-1))
            else:
                loss = outputs.loss
            
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / max(1, num_batches)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = PIIDataset(args.train, tokenizer, LABELS, max_length=args.max_length, is_train=True)
    
    # Optionally combine train and dev for very small datasets
    if args.combine_train_dev:
        print("Combining train and dev sets for training...")
        dev_ds_train = PIIDataset(args.dev, tokenizer, LABELS, max_length=args.max_length, is_train=True)
        # Combine datasets
        train_ds.items.extend(dev_ds_train.items)
        print(f"Combined dataset size: {len(train_ds)} examples")
    
    # For very small datasets, repeat data multiple times to increase effective dataset size
    if len(train_ds) < 10:
        repeat_factor = max(1, 10 // len(train_ds))
        if repeat_factor > 1:
            original_items = train_ds.items.copy()
            train_ds.items = original_items * repeat_factor
            print(f"Repeated training data {repeat_factor}x times. Effective size: {len(train_ds)} examples")
    
    dev_ds = PIIDataset(args.dev, tokenizer, LABELS, max_length=args.max_length, is_train=False)
    
    # Calculate class weights to handle label imbalance
    label_counts = Counter()
    for item in train_ds:
        for label_id in item["labels"]:
            if label_id != -100:  # Ignore padding
                label_counts[label_id] += 1
    
    # Compute class weights (inverse frequency)
    total_samples = sum(label_counts.values())
    num_classes = len(LABELS)
    class_weights = torch.ones(num_classes, device=args.device)
    
    for label_id, count in label_counts.items():
        if count > 0:
            # Inverse frequency weighting: more frequent = lower weight
            # Use sqrt to make weights less aggressive
            class_weights[label_id] = np.sqrt(total_samples / (num_classes * count))
    
    # Normalize weights to prevent extreme values
    class_weights = class_weights / class_weights.sum() * num_classes
    # Cap maximum weight to prevent over-weighting
    max_weight = 5.0
    class_weights = torch.clamp(class_weights, min=0.1, max=max_weight)
    print(f"Class weights computed: O={class_weights[LABEL2ID['O']]:.3f}, avg_entity={class_weights[1:].mean():.3f}")

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )
    
    dev_dl = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )

    model = create_model(args.model_name)
    model.to(args.device)
    model.train()

    # Better optimizer with weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # Use AdamW with lower learning rate for fine-tuning
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    
    total_steps = len(train_dl) * args.epochs
    num_warmup_steps = max(1, int(args.warmup_ratio * total_steps))
    # Use linear schedule with warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )

    best_dev_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        running_loss = 0.0
        model.train()
        
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = torch.tensor(batch["input_ids"], device=args.device)
            attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
            labels = torch.tensor(batch["labels"], device=args.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            # Use weighted loss to handle class imbalance
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
            loss = loss_fct(logits.view(-1, len(LABELS)), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_dl))
        
        # Evaluate on dev set
        dev_loss = evaluate(model, dev_dl, args.device, class_weights)
        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}, dev_loss={dev_loss:.4f}")
        
        # Early stopping
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            patience_counter = 0
            # Save best model
            model.save_pretrained(args.out_dir)
            tokenizer.save_pretrained(args.out_dir)
            print(f"  -> New best model saved (dev_loss={dev_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping after {epoch+1} epochs (patience={args.patience})")
                break

    print(f"Training completed. Best dev loss: {best_dev_loss:.4f}")
    print(f"Saved model + tokenizer to {args.out_dir}")


if __name__ == "__main__":
    main()
