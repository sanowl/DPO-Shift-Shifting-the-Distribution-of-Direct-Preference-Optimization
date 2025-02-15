# DPO-Shift: Shifting the Distribution of Direct Preference Optimization

This project implements two training regimes for language models:

## 1. Supervised Fine-Tuning (SFT)
- **Objective**: Minimize the negative log-likelihood.
- **Loss**: For a given input sample with prompt $p$ and response $r$, the loss is computed as:
  
  $$
  \mathcal{L}_{\text{SFT}} = -\log P(r \mid p)
  $$

- **Training**: Uses the SFT loop to optimize model parameters with gradient descent and an optional scheduler:
  
  - Model update: 
    $$
    \theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}_{\text{SFT}}
    $$

## 2. Pairwise Optimization (PO)
- **Objective**: Use paired preferences where one response is chosen over another.
- **Formulation**: Let 
  $$
  \Delta_{\text{chosen}} = \log P_{\text{policy}}(r_{\text{chosen}} \mid p) - \log P_{\text{ref}}(r_{\text{chosen}} \mid p)
  $$
  and 
  $$
  \Delta_{\text{rejected}} = \log P_{\text{policy}}(r_{\text{rejected}} \mid p) - \log P_{\text{ref}}(r_{\text{rejected}} \mid p)
  $$
  
- **Lambda Scheduling**: A strategy function computes 
  $$
  \lambda = f(\text{step}) \quad \text{with strategies such as fixed, linear increase, or linear decrease.}
  $$
  
- **Loss**: The pairwise loss is defined as:
  
  $$
  \mathcal{L}_{\text{PO}} = -\log \sigma\left(\beta \left(\Delta_{\text{chosen}} - \lambda \Delta_{\text{rejected}}\right)\right)
  $$
  
  where $\beta$ is a scaling factor and $\sigma$ is the sigmoid function.

## Additional Components
- **Dataset**: The `PreferenceDataset` class prepares tokenized inputs for both chosen and rejected responses.
- **Checkpoints**: The project saves model checkpoints containing optimizer and scheduler states.
- **Logging and Evaluation**: Uses Weights & Biases for logging metrics such as training loss, validation loss, margins, and accuracy.

## Running the Project
- Start training by executing the `main()` function in [main.py](./main.py).
- Ensure that necessary libraries (e.g., PyTorch, Transformers) are installed.

This README provides a mathematical intuition behind each method used in the code.
