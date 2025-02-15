#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
import numpy as np
from torch.utils.data import Dataset, DataLoader
import wandb
import os
from pathlib import Path

class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        prompt = sample["prompt"]
        chosen = sample["chosen"] if "chosen" in sample else sample["response"]
        rejected = sample.get("rejected", None)

        chosen_tokens = self.tokenizer(prompt + chosen, 
                                     max_length=self.max_length, 
                                     truncation=True, 
                                     padding="max_length", 
                                     return_tensors="pt")
        
        prompt_tokens = self.tokenizer(prompt, 
                                     max_length=self.max_length, 
                                     truncation=True, 
                                     return_tensors="pt")

        chosen_labels = chosen_tokens.input_ids.clone()
        chosen_labels[0, :prompt_tokens.input_ids.size(1)] = -100

        item = {
            "chosen_input_ids": chosen_tokens.input_ids[0],
            "chosen_attention_mask": chosen_tokens.attention_mask[0],
            "chosen_labels": chosen_labels[0],
            "prompt_length": prompt_tokens.input_ids.size(1)
        }

        if rejected is not None:
            rejected_tokens = self.tokenizer(prompt + rejected, 
                                           max_length=self.max_length, 
                                           truncation=True, 
                                           padding="max_length", 
                                           return_tensors="pt")
            
            rejected_labels = rejected_tokens.input_ids.clone()
            rejected_labels[0, :prompt_tokens.input_ids.size(1)] = -100

            item.update({
                "rejected_input_ids": rejected_tokens.input_ids[0],
                "rejected_attention_mask": rejected_tokens.attention_mask[0],
                "rejected_labels": rejected_labels[0]
            })

        return item

def get_f_lambda(strategy, step, total_steps, lambda_min, lambda_max):
    if strategy == "fixed":
        return lambda_max
    elif strategy == "linear_increase":
        return step / total_steps * (lambda_max - lambda_min) + lambda_min
    elif strategy == "linear_decrease":
        return step / total_steps * (lambda_min - lambda_max) + lambda_max
    else:
        raise ValueError("Unknown strategy")

def compute_logp(model, input_ids, attention_mask, labels, grad=True):
    if grad:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    else:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    response_length = (labels != -100).sum().item()
    logp = -outputs.loss * response_length
    return logp

def save_checkpoint(model, optimizer, scheduler, epoch, step, metrics, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'step': step,
        'metrics': metrics
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, scheduler, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['step'], checkpoint['metrics']

def evaluate_metrics(model, reference_model, dataloader, device, f_lambda_val):
    model.eval()
    chosen_logs = []
    margins = []
    
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        logp_chosen_policy = compute_logp(model, batch['chosen_input_ids'], 
                                        batch['chosen_attention_mask'], 
                                        batch['chosen_labels'], grad=False)
        logp_rejected_policy = compute_logp(model, batch['rejected_input_ids'], 
                                          batch['rejected_attention_mask'], 
                                          batch['rejected_labels'], grad=False)
        
        logp_chosen_ref = compute_logp(reference_model, batch['chosen_input_ids'], 
                                     batch['chosen_attention_mask'], 
                                     batch['chosen_labels'], grad=False)
        logp_rejected_ref = compute_logp(reference_model, batch['rejected_input_ids'], 
                                       batch['rejected_attention_mask'], 
                                       batch['rejected_labels'], grad=False)

        log_ratio_chosen = logp_chosen_policy - logp_chosen_ref
        log_ratio_rejected = logp_rejected_policy - logp_rejected_ref
        margin = log_ratio_chosen - f_lambda_val * log_ratio_rejected

        chosen_logs.extend(logp_chosen_policy.cpu().numpy())
        margins.extend(margin.cpu().numpy())

    return np.mean(chosen_logs), np.mean(margins)

def train_SFT(model, train_dataloader, val_dataloader, epochs, optimizer, scheduler, device, 
              max_grad_norm=1.0, checkpoint_dir="checkpoints", log_interval=100):
    model.train()
    step = 0
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(input_ids=batch['chosen_input_ids'],
                          attention_mask=batch['chosen_attention_mask'],
                          labels=batch['chosen_labels'])
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            step += 1
            
            if step % log_interval == 0:
                wandb.log({
                    "sft_train_loss": loss.item(),
                    "epoch": epoch,
                    "step": step,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
        
        val_loss = validate_SFT(model, val_dataloader, device)
        wandb.log({
            "sft_val_loss": val_loss,
            "epoch": epoch
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, step,
                          {"val_loss": val_loss},
                          os.path.join(checkpoint_dir, f"sft_best.pt"))
        
        save_checkpoint(model, optimizer, scheduler, epoch, step,
                       {"val_loss": val_loss},
                       os.path.join(checkpoint_dir, f"sft_latest.pt"))

def train_PO(policy_model, reference_model, train_dataloader, val_dataloader, epochs, 
             optimizer, scheduler, strategy, lambda_min, lambda_max, device, beta=1.0,
             max_grad_norm=1.0, checkpoint_dir="checkpoints", log_interval=100):
    policy_model.train()
    step = 0
    total_steps = epochs * len(train_dataloader)
    best_val_margin = float('-inf')
    
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            logp_chosen_policy = compute_logp(policy_model, batch['chosen_input_ids'],
                                            batch['chosen_attention_mask'],
                                            batch['chosen_labels'], grad=True)
            logp_rejected_policy = compute_logp(policy_model, batch['rejected_input_ids'],
                                              batch['rejected_attention_mask'],
                                              batch['rejected_labels'], grad=True)
            
            with torch.no_grad():
                logp_chosen_ref = compute_logp(reference_model, batch['chosen_input_ids'],
                                             batch['chosen_attention_mask'],
                                             batch['chosen_labels'], grad=False)
                logp_rejected_ref = compute_logp(reference_model, batch['rejected_input_ids'],
                                               batch['rejected_attention_mask'],
                                               batch['rejected_labels'], grad=False)
            
            log_ratio_chosen = logp_chosen_policy - logp_chosen_ref
            log_ratio_rejected = logp_rejected_policy - logp_rejected_ref
            
            current_f_lambda = get_f_lambda(strategy, step, total_steps, lambda_min, lambda_max)
            diff = beta * (log_ratio_chosen - current_f_lambda * log_ratio_rejected)
            loss = -torch.log(torch.sigmoid(diff) + 1e-7)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            step += 1
            
            if step % log_interval == 0:
                with torch.no_grad():
                    acc = (diff > 0).float().mean()
                wandb.log({
                    "po_train_loss": loss.item(),
                    "po_train_acc": acc.item(),
                    "f_lambda": current_f_lambda,
                    "epoch": epoch,
                    "step": step,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
        
        val_chosen_logp, val_margin = evaluate_metrics(policy_model, reference_model,
                                                     val_dataloader, device, current_f_lambda)
        wandb.log({
            "val_chosen_logp": val_chosen_logp,
            "val_margin": val_margin,
            "epoch": epoch
        })
        
        if val_margin > best_val_margin:
            best_val_margin = val_margin
            save_checkpoint(policy_model, optimizer, scheduler, epoch, step,
                          {"val_margin": val_margin},
                          os.path.join(checkpoint_dir, f"po_best.pt"))
        
        save_checkpoint(policy_model, optimizer, scheduler, epoch, step,
                       {"val_margin": val_margin},
                       os.path.join(checkpoint_dir, f"po_latest.pt"))

def validate_SFT(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['chosen_input_ids'],
                          attention_mask=batch['chosen_attention_mask'],
                          labels=batch['chosen_labels'])
            total_loss += outputs.loss.item()
    return total_loss / len(dataloader)

def main():
    wandb.init(project="dpo-shift")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    sft_dataset = [
        {"prompt": "What is the capital of France? ", "response": "Paris."},
        {"prompt": "Who wrote '1984'? ", "response": "George Orwell."},
        {"prompt": "What is the largest planet in our Solar System? ", "response": "Jupiter."}
    ]
    
    train_size = int(0.8 * len(sft_dataset))
    train_dataset = PreferenceDataset(sft_dataset[:train_size], tokenizer)
    val_dataset = PreferenceDataset(sft_dataset[train_size:], tokenizer)
    
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4)
    
    sft_epochs = 3
    sft_optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    num_training_steps = sft_epochs * len(train_dataloader)
    sft_scheduler = get_linear_schedule_with_warmup(
        sft_optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps
    )
    
    os.makedirs("checkpoints", exist_ok=True)
    
    train_SFT(model, train_dataloader, val_dataloader, sft_epochs,
              sft_optimizer, sft_scheduler, device)
    
    reference_model = copy.deepcopy(model)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    
    po_dataset = [
        {"prompt": "What is the capital of France? ",
         "chosen": "Paris.", "rejected": "Lyon."},
        {"prompt": "Who wrote '1984'? ",
         "chosen": "George Orwell.", "rejected": "Aldous Huxley."},
        {"prompt": "What is the largest planet in our Solar System? ",
         "chosen": "Jupiter.", "rejected": "Saturn."}
    ]
    
    train_dataset = PreferenceDataset(po_dataset[:train_size], tokenizer)
    val_dataset = PreferenceDataset(po_dataset[train_size:], tokenizer)
    
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4)
    
    po_epochs = 3
    po_optimizer = optim.AdamW(model.parameters(), lr=1e-6)
    num_training_steps = po_epochs * len(train_dataloader)
    po_scheduler = get_linear_schedule_with_warmup(
        po_optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps
    )
    
    strategy = "fixed"
    lambda_min = 0.75
    lambda_max = 0.95

    train_PO(model, reference_model, train_dataloader, val_dataloader, po_epochs,
             po_optimizer, po_scheduler, strategy, lambda_min, lambda_max, device)

    final_chosen, final_margin = evaluate_metrics(model, reference_model, val_dataloader, device, lambda_max)
    wandb.log({
        "final_chosen_logp": final_chosen,
        "final_margin": final_margin
    })

    save_dir = "dpo_shift_trained_model"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    wandb.finish()

if __name__ == "__main__":
    main()