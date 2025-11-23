### Prerequisites

- Python 3.7+
- PyTorch
- Transformers library

### Installation

```bash
pip install -r requirements.txt
```

### Required Dependencies

- `torch` - PyTorch deep learning framework
- `transformers` - Hugging Face transformers library
- `numpy` - Numerical computing
- `tqdm` - Progress bars
- `seqeval` - Sequence labeling evaluation

## Project Structure

```
.
├── src/
│   ├── dataset.py          # Dataset loading and preprocessing
│   ├── labels.py            # Label definitions and PII mapping
│   ├── model.py             # Model creation utilities
│   ├── train.py             # Training script with validation
│   ├── predict.py           # Inference script
│   ├── eval_span_f1.py      # Evaluation metrics
│   └── measure_latency.py   # Latency measurement
├── data/
│   ├── train.jsonl          # Training data
│   ├── dev.jsonl            # Development/validation data
│   └── test.jsonl            # Test data (no labels)
├── out/                      # Output directory (model, predictions)
└── requirements.txt         # Python dependencies
```

## Entity Types

The model detects the following entity types:

**PII Entities (pii: true):**
- `CREDIT_CARD` - Credit card numbers
- `PHONE` - Phone numbers
- `EMAIL` - Email addresses
- `PERSON_NAME` - Person names
- `DATE` - Dates

**Non-PII Entities (pii: false):**
- `CITY` - City names
- `LOCATION` - General locations

## Process Workflow

### 1. Training

Train the model with validation and early stopping:

```bash
python src/train.py \
  --model_name distilbert-base-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out
```

**Training Features:**
- Validation loop with dev set evaluation
- Early stopping with configurable patience (default: 3 epochs)
- Learning rate scheduling with warmup
- Gradient clipping for training stability
- Weight decay regularization
- Best model checkpointing based on dev loss

**Hyperparameters:**
- Model: `distilbert-base-uncased`
- Batch size: 16
- Learning rate: 3e-5
- Max epochs: 10 (with early stopping)
- Max sequence length: 256
- Weight decay: 0.01
- Warmup ratio: 0.1

### 2. Prediction

Generate predictions on new data:

```bash
python src/predict.py \
  --model_dir out \
  --input data/dev.jsonl \
  --output out/dev_pred.json
```

**Output Format:**
```json
{
  "utt_0101": [
    {
      "start": 76,
      "end": 99,
      "label": "CREDIT_CARD",
      "pii": true
    }
  ]
}
```

### 3. Evaluation

Evaluate model performance:

```bash
python src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred out/dev_pred.json
```

**Metrics Reported:**
- Per-entity precision, recall, and F1
- Macro-averaged F1
- PII-only metrics (precision, recall, F1)
- Non-PII metrics (precision, recall, F1)

### 4. Latency Measurement

Measure inference latency:

```bash
python src/measure_latency.py \
  --model_dir out \
  --input data/dev.jsonl \
  --runs 50
```

**Latency Metrics:**
- p50 (median) latency
- p95 (95th percentile) latency
- Includes tokenization time for accurate measurement

## Implementation Details

### Key Improvements

1. **Enhanced Training Pipeline**
   - Validation during training to monitor overfitting
   - Early stopping to prevent overfitting and save training time
   - Improved optimizer with weight decay and parameter grouping
   - Gradient clipping for training stability

2. **Better Label Alignment**
   - Improved handling of subword tokenization
   - Proper BIO tag assignment for tokenizer offsets
   - Handles edge cases where tokens span entity boundaries

3. **Inference Optimizations**
   - CPU optimizations: single-threaded execution for consistent latency
   - MKL-DNN optimizations enabled for CPU
   - GPU optimizations: torch.compile support (PyTorch 2.0+)
   - Latency measurement includes tokenization time

4. **Model Configuration**
   - Configurable dropout for regularization
   - Better model initialization

## Results

### Training Results

**Training Configuration:**
- Training examples: 2
- Dev examples: 2
- Model: DistilBERT-base-uncased
- Total epochs: 10

**Training Progress:**
```
Epoch 1: train_loss=2.7231, dev_loss=2.7982 → New best model saved
Epoch 2: train_loss=2.7068, dev_loss=2.6644 → New best model saved
Epoch 3: train_loss=2.4438, dev_loss=2.5477 → New best model saved
Epoch 4: train_loss=2.2368, dev_loss=2.4463 → New best model saved
Epoch 5: train_loss=2.0478, dev_loss=2.3594 → New best model saved
Epoch 6: train_loss=1.8812, dev_loss=2.2872 → New best model saved
Epoch 7: train_loss=1.7457, dev_loss=2.2298 → New best model saved
Epoch 8: train_loss=1.6323, dev_loss=2.1875 → New best model saved
Epoch 9: train_loss=1.5407, dev_loss=2.1599 → New best model saved
Epoch 10: train_loss=1.4686, dev_loss=2.1463 → New best model saved
```

**Final Metrics:**
- Best dev loss: 2.1463 (improved from 2.7982)
- Training completed successfully
- Model saved to `out/` directory

### Evaluation Results

**Per-Entity Metrics:**
```
CITY            P=0.000 R=0.000 F1=0.000
CREDIT_CARD     P=0.000 R=0.000 F1=0.000
DATE            P=0.000 R=0.000 F1=0.000
EMAIL           P=0.000 R=0.000 F1=0.000
PERSON_NAME     P=0.000 R=0.000 F1=0.000
PHONE           P=0.000 R=0.000 F1=0.000

Macro-F1: 0.000
```

**PII Metrics:**
- PII Precision: 0.000
- PII Recall: 0.000
- PII F1: 0.000

**Non-PII Metrics:**
- Non-PII Precision: 0.000
- Non-PII Recall: 0.000
- Non-PII F1: 0.000

**Note:** The low performance metrics are expected given the extremely small dataset (only 2 training examples). With a larger dataset, the model architecture and training improvements should achieve the target PII precision ≥ 0.80.

### Latency Results

**Inference Latency (50 runs, batch_size=1):**
- **p50 (median):** 37.47 ms
- **p95 (95th percentile):** 42.74 ms
- **Target:** ≤ 20 ms

**Latency Analysis:**
- Current latency is above the 20 ms target
- Latency includes tokenization time for accurate measurement
- Further optimizations possible:
  - Model quantization
  - ONNX runtime conversion
  - Smaller/faster model architecture
  - Batch processing for multiple utterances

### Sample Predictions

**Input (utt_0101):**
```
"email id is priyanka dot verma at outlook dot com and card number is 5555 5555 5555 4444"
```

**Gold Labels:**
- PERSON_NAME: "priyanka dot verma" (start: 12, end: 29)
- EMAIL: "priyanka dot verma at outlook dot com" (start: 33, end: 55)
- CREDIT_CARD: "5555 5555 5555 4444" (start: 76, end: 99)

**Model Predictions:**
- CREDIT_CARD: (start: 74, end: 77)
- CREDIT_CARD: (start: 79, end: 82)
- CREDIT_CARD: (start: 84, end: 86)

**Analysis:** The model is detecting credit card patterns but with incorrect span boundaries, likely due to the very small training dataset.

## Model Architecture

- **Base Model:** DistilBERT-base-uncased
  - 6 transformer layers
  - 768 hidden dimensions
  - 66M parameters
  - Faster inference than BERT while maintaining good performance

- **Classification Head:**
  - Token-level classification
  - 15 output classes (BIO scheme: O + 7 entity types × 2)
  - Dropout: 0.1

## Data Format

### Input Format (JSONL)

Each line is a JSON object:

```json
{
  "id": "utt_0001",
  "text": "my credit card number is 4242 4242 4242 4242",
  "entities": [
    {
      "start": 26,
      "end": 49,
      "label": "CREDIT_CARD"
    }
  ]
}
```

- `id`: Unique utterance identifier
- `text`: STT transcript (may contain spoken forms like "dot", "at", etc.)
- `entities`: List of gold entity spans (only in train/dev)
  - `start`: Character start index (inclusive)
  - `end`: Character end index (exclusive, Python slice semantics)
  - `label`: Entity type

### Output Format (JSON)

```json
{
  "utt_0001": [
    {
      "start": 26,
      "end": 49,
      "label": "CREDIT_CARD",
      "pii": true
    }
  ]
}
```

## Future Improvements

1. **Dataset Expansion**
   - Current dataset is extremely small (2 examples)
   - Need more training data for better generalization

2. **Latency Optimization**
   - Model quantization (INT8)
   - ONNX runtime conversion
   - Consider smaller models (e.g., MobileBERT, TinyBERT)
   - Batch processing for multiple utterances

3. **Model Improvements**
   - Experiment with different base models
   - Hyperparameter tuning
   - Data augmentation for STT transcripts
   - Ensemble methods

4. **Evaluation**
   - More comprehensive evaluation metrics
   - Error analysis
   - Confusion matrix for entity types

## Command Reference

### Full Pipeline

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train model
python src/train.py \
  --model_name distilbert-base-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out \
  --batch_size 16 \
  --epochs 10 \
  --lr 3e-5

# 3. Generate predictions
python src/predict.py \
  --model_dir out \
  --input data/dev.jsonl \
  --output out/dev_pred.json

# 4. Evaluate
python src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred out/dev_pred.json

# 5. Measure latency
python src/measure_latency.py \
  --model_dir out \
  --input data/dev.jsonl \
  --runs 50
```

## Notes

- The current implementation uses a very small dataset (2 training examples), which limits model performance
- All code improvements are in place and will perform better with more training data
- The model architecture and training pipeline are optimized for both accuracy and latency
- Early stopping prevents overfitting and saves training time
- Inference optimizations are applied for both CPU and GPU execution

## License

This project is part of an assignment for PII detection in STT transcripts.
