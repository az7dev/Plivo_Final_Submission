#!/bin/bash

set -e

echo "=== Setting up Miniconda && Activating Environment ==="
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/bin/activate

echo "=== do torch install && install hf_transfer ==="
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
python -m pip install ninja
pip install hf_transfer

echo "=== Cloning Repository ==="
git clone https://github.com/az7dev/Plivo_Final_Submission.git
cd Plivo_Final_Submission

echo "=== Installing Dependencies ==="
pip install -r requirements.txt


echo "=== Training Model ==="
python src/train.py \
    --model_name google/bert_uncased_L-4_H-256_A-4 \
    --train data/train.jsonl \
    --dev data/dev.jsonl \
    --out_dir out

echo "=== Generating Predictions ==="
python src/predict.py \
    --model_dir out \
    --input data/dev.jsonl \
    --output out/dev_pred.json

echo "=== Evaluating Model (F1, PII Precision, Span F1) ==="
python src/eval_span_f1.py \
    --gold data/dev.jsonl \
    --pred out/dev_pred.json

echo "=== Measuring Latency (p50, p95) ==="
python src/measure_latency.py \
    --model_dir out \
    --input data/dev.jsonl \
    --runs 50

echo "=== Done ==="
