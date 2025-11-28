import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import AdamW

def run_sft(model, dataset, config, device):
    print("\n=== Starting Step 1: Supervised Fine-Tuning (SFT) ===")
    model.train()
    optimizer = AdamW(model.parameters(), lr=config.sft_lr)
    dataloader = DataLoader(dataset, batch_size=config.sft_batch_size, shuffle=True)
    
    for epoch in range(config.sft_epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"SFT Epoch {epoch+1}")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
    print("=== SFT Completed ===\n")
    return model