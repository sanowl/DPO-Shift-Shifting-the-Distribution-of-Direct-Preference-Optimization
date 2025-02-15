# DPO-Shift: Shifting the Distribution of Direct Preference Optimization

An implementation of Direct Preference Optimization (DPO) with dynamic preference shifting.

## Overview

This project trains language models using two stages:

### 1. Supervised Fine-Tuning (SFT)

The SFT stage trains the model to generate responses by minimizing:

![equation](https://latex.codecogs.com/png.latex?%5Cmathcal%7BL%7D_%7B%5Ctext%7BSFT%7D%7D%20%3D%20-%5Clog%20P%28r%20%5Cmid%20p%29)

Where:
- `p` is the prompt
- `r` is the desired response
- `P(r|p)` is the model's probability of generating response `r` given prompt `p`

### 2. DPO with Lambda Shifting

The DPO stage uses paired responses (chosen vs rejected) to optimize preferences.

For each pair, we compute:

![equation](https://latex.codecogs.com/png.latex?%5CDelta_%7Bchosen%7D%20%3D%20%5Clog%20%5Cfrac%7BP_%7Bpolicy%7D%28r_%7Bchosen%7D%20%5Cmid%20p%29%7D%7BP_%7Bref%7D%28r_%7Bchosen%7D%20%5Cmid%20p%29%7D)

![equation](https://latex.codecogs.com/png.latex?%5CDelta_%7Brejected%7D%20%3D%20%5Clog%20%5Cfrac%7BP_%7Bpolicy%7D%28r_%7Brejected%7D%20%5Cmid%20p%29%7D%7BP_%7Bref%7D%28r_%7Brejected%7D%20%5Cmid%20p%29%7D)

The loss function is:

![equation](https://latex.codecogs.com/png.latex?%5Cmathcal%7BL%7D_%7BDPO%7D%20%3D%20-%5Clog%20%5Csigma%28%5Cbeta%28%5CDelta_%7Bchosen%7D%20-%20%5Clambda%20%5CDelta_%7Brejected%7D%29%29)

Where:
- `λ` (lambda) is dynamically adjusted using different strategies
- `β` is a temperature parameter
- `σ` is the sigmoid function

## Key Features

- Three lambda scheduling strategies:
  - Fixed: `λ = constant`
  - Linear increase: `λ = λ_min + t(λ_max - λ_min)`
  - Linear decrease: `λ = λ_max - t(λ_max - λ_min)`
  where `t` is the normalized training step

- Automatic checkpointing
- W&B logging for:
  - Training/validation losses
  - Chosen/rejected margins
  - Lambda values
  - Accuracy metrics

## Usage

1. Install dependencies:
```bash
pip install torch transformers wandb
```

2. Run training:
```bash
python main.py
```

## Implementation Details

- Base model: GPT-2
- Training parameters:
  - SFT learning rate: 1e-5
  - DPO learning rate: 1e-6
  - Batch size: 4
  - Gradient clipping: 1.0
  - Warmup steps: 10% of total steps

## Results

The model improves preference alignment while maintaining language capabilities through:
1. Initial SFT to learn basic response generation
2. DPO with shifting to fine-tune preference alignment
