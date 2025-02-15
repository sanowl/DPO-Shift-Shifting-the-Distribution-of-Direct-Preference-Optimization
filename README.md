# DPO-Shift: Shifting the Distribution of Direct Preference Optimization

An implementation of Direct Preference Optimization (DPO) with dynamic preference shifting.

## Overview

This project trains language models using two stages:

### 1. Supervised Fine-Tuning (SFT)

The SFT stage trains the model to generate responses by minimizing:

$$ \mathcal{L}_\text{SFT} = -\log P(r|p) $$

Where:
- $p$ is the prompt
- $r$ is the desired response
- $P(r|p)$ is the model's probability of generating response $r$ given prompt $p$

### 2. DPO with Lambda Shifting

The DPO stage uses paired responses (chosen vs rejected) to optimize preferences.

For each pair, we compute:

$$ \Delta_\text{chosen} = \log \frac{P_\text{policy}(r_\text{chosen}|p)}{P_\text{ref}(r_\text{chosen}|p)} $$

$$ \Delta_\text{rejected} = \log \frac{P_\text{policy}(r_\text{rejected}|p)}{P_\text{ref}(r_\text{rejected}|p)} $$

The loss function is:

$$ \mathcal{L}_\text{DPO} = -\log \sigma(\beta(\Delta_\text{chosen} - \lambda \Delta_\text{rejected})) $$

Where:
- $\lambda$ (lambda) is dynamically adjusted using different strategies
- $\beta$ is a temperature parameter
- $\sigma$ is the sigmoid function

## Lambda Scheduling Strategies

1. Fixed:
   $$ \lambda = \text{constant} $$

2. Linear increase:
   $$ \lambda = \lambda_\text{min} + t(\lambda_\text{max} - \lambda_\text{min}) $$

3. Linear decrease:
   $$ \lambda = \lambda_\text{max} - t(\lambda_\text{max} - \lambda_\text{min}) $$

where $t$ is the normalized training step $(0 \leq t \leq 1)$

## Key Features

- Dynamic lambda scheduling
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
