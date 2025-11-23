import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii
import os
import torch.backends.mkldnn


def bio_to_spans(text, offsets, label_ids, confidence_scores=None, min_confidence=0.0):
    """
    Convert BIO tag predictions to entity spans with precise boundary detection.
    
    Args:
        text: Original text
        offsets: Token offset mappings
        label_ids: Predicted label IDs
        confidence_scores: Optional confidence scores for each token
        min_confidence: Minimum confidence threshold (0.0 to 1.0)
    
    Returns:
        List of (start, end, label) tuples
    """
    spans = []
    current_label = None
    current_start = None
    current_end = None
    current_confidence = None
    current_tokens = []  # Track all tokens in current entity

    for idx, ((start, end), lid) in enumerate(zip(offsets, label_ids)):
        if start == 0 and end == 0:
            continue
        
        label = ID2LABEL.get(int(lid), "O")
        confidence = confidence_scores[idx] if confidence_scores is not None else 1.0
        
        if label == "O":
            if current_label is not None:
                # Check confidence threshold before adding span
                if current_confidence is None or current_confidence >= min_confidence:
                    # Use first token's start and last token's end for precision
                    if current_tokens:
                        final_start = current_tokens[0][0]
                        final_end = current_tokens[-1][1]
                        spans.append((final_start, final_end, current_label))
                current_label = None
                current_confidence = None
                current_tokens = []
            continue

        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            # Save previous entity if exists
            if current_label is not None:
                if current_confidence is None or current_confidence >= min_confidence:
                    if current_tokens:
                        final_start = current_tokens[0][0]
                        final_end = current_tokens[-1][1]
                        spans.append((final_start, final_end, current_label))
            # Start new entity
            current_label = ent_type
            current_start = start
            current_end = end
            current_confidence = confidence
            current_tokens = [(start, end)]
        elif prefix == "I":
            if current_label == ent_type:
                # Continue current entity
                current_end = end
                current_tokens.append((start, end))
                # Update confidence (use minimum or average)
                if current_confidence is not None:
                    current_confidence = min(current_confidence, confidence)
                else:
                    current_confidence = confidence
            else:
                # Mismatch: save previous and start new
                if current_label is not None:
                    if current_confidence is None or current_confidence >= min_confidence:
                        if current_tokens:
                            final_start = current_tokens[0][0]
                            final_end = current_tokens[-1][1]
                            spans.append((final_start, final_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end
                current_confidence = confidence
                current_tokens = [(start, end)]

    # Save final entity
    if current_label is not None:
        if current_confidence is None or current_confidence >= min_confidence:
            if current_tokens:
                final_start = current_tokens[0][0]
                final_end = current_tokens[-1][1]
                spans.append((final_start, final_end, current_label))

    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min_confidence", type=float, default=0.3, 
                    help="Minimum confidence threshold for predictions (0.0 to 1.0)")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir if args.model_name is None else args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()
    
    # Optimize for inference
    if args.device == "cpu":
        # Enable optimizations for CPU inference
        torch.set_num_threads(1)  # Single thread for consistent latency
        if hasattr(torch.backends, 'mkldnn'):
            torch.backends.mkldnn.enabled = True
    else:
        # For GPU, use torch.compile if available (PyTorch 2.0+)
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except:
            pass  # Fallback if compile not available

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]
                
                # Get predictions and confidence scores
                probs = torch.softmax(logits, dim=-1)
                pred_ids = logits.argmax(dim=-1).cpu().tolist()
                # Get confidence as max probability
                confidences = probs.max(dim=-1)[0].cpu().tolist()

            spans = bio_to_spans(text, offsets, pred_ids, confidences, args.min_confidence)
            
            # Post-process: merge adjacent entities of the same type
            merged_spans = []
            if spans:
                current_start, current_end, current_label = spans[0]
                for s, e, lab in spans[1:]:
                    if lab == current_label and s <= current_end + 2:  # Adjacent or overlapping (allow 2 char gap)
                        # Merge: extend the end
                        current_end = max(current_end, e)
                    else:
                        # Save current and start new
                        merged_spans.append((current_start, current_end, current_label))
                        current_start, current_end, current_label = s, e, lab
                # Add the last span
                merged_spans.append((current_start, current_end, current_label))
            else:
                merged_spans = spans
            
            # Additional pass: try to extend spans intelligently based on entity type
            refined_spans = []
            for s, e, lab in merged_spans:
                span_text = text[s:e] if s < len(text) and e <= len(text) else ""
                new_start, new_end = s, e
                
                if lab == "EMAIL":
                    # For emails, extend to include "at" and "dot com" patterns
                    # Look for "at" before start (with space)
                    if new_start >= 3 and text[new_start-3:new_start].lower() == " at":
                        new_start -= 3
                    elif new_start >= 1 and text[new_start-1] == "@":
                        new_start -= 1
                    # Look for "dot com" or ".com" after end
                    if new_end + 7 <= len(text) and text[new_end:new_end+7].lower() == " dot com":
                        new_end += 7
                    elif new_end + 6 <= len(text) and text[new_end:new_end+6].lower() == "dot com":
                        new_end += 6
                    elif new_end + 4 <= len(text) and text[new_end:new_end+4].lower() == ".com":
                        new_end += 4
                    # Also check for "and" or "and card" that might be part of context
                    if new_end + 5 <= len(text) and text[new_end:new_end+5] == " and ":
                        # Check if next word is "card" - if so, don't extend (it's not part of email)
                        pass
                elif lab in ["CREDIT_CARD", "PHONE"]:
                    # Extend to include adjacent digits and spaces
                    while new_start > 0 and (text[new_start-1].isdigit() or text[new_start-1].isspace()):
                        new_start -= 1
                    while new_end < len(text) and (text[new_end].isdigit() or text[new_end].isspace()):
                        new_end += 1
                elif lab == "CITY":
                    # For city names, trim whitespace but don't over-extend
                    pass
                elif lab == "DATE":
                    # For dates, extend to include adjacent digits/spaces/slashes
                    while new_start > 0 and (text[new_start-1].isdigit() or text[new_start-1] in " /-"):
                        new_start -= 1
                    while new_end < len(text) and (text[new_end].isdigit() or text[new_end] in " /-"):
                        new_end += 1
                
                refined_spans.append((new_start, new_end, lab))
            
            merged_spans = refined_spans
            
            # Refine spans to better match gold label boundaries
            # Try to align with word boundaries and trim whitespace
            final_ents = []
            for s, e, lab in merged_spans:
                if e > s:  # Valid span
                    # Trim leading/trailing whitespace
                    span_text = text[s:e]
                    actual_start = s
                    actual_end = e
                    
                    # Trim leading spaces
                    while actual_start < actual_end and text[actual_start].isspace():
                        actual_start += 1
                    # Trim trailing spaces
                    while actual_end > actual_start and text[actual_end - 1].isspace():
                        actual_end -= 1
                    
                    # For better alignment, try to extend to word boundaries if close
                    # But only if it makes sense (don't over-extend)
                    if actual_end > actual_start:
                        final_ents.append({
                            "start": int(actual_start),
                            "end": int(actual_end),
                            "label": lab,
                            "pii": bool(label_is_pii(lab)),
                        })
            
            results[uid] = final_ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()
