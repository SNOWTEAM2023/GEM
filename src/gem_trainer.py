import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import AdamW
from .entropy_scorer import EntropyScorer

class GEMTrainer:
    def __init__(self, model, tokenizer, config, device):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.scorer = EntropyScorer(config, tokenizer)
        self.optimizer = AdamW(self.model.parameters(), lr=config.sega_lr)

    def train_sega(self, dataset):
        print("=== Starting Step 2: GEM Optimization (SEGA) ===")
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        self.model.train()
        
        for epoch in range(self.config.sega_epochs):
            pbar = tqdm(dataloader, desc=f"SEGA Epoch {epoch+1}")
            optimizer_step = 0
            accumulated_loss = 0
            
            for step, prompt_batch in enumerate(pbar):
                prompt_str = prompt_batch[0] # Tuple from dataloader
                
                # --- A. Cognitive Filtering: 生成候选 (Inference Mode) ---
                inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.device)
                prompt_len = inputs.input_ids.shape[1]
                
                with torch.no_grad():
                    # 生成 K 个 CoT
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=self.config.temperature,
                        num_return_sequences=self.config.k_candidates,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # --- B. Scoring & Advantage Computation ---
                gem_scores = []
                log_probs_sum = []
                
                
                forward_outputs = self.model(outputs)
                all_logits = forward_outputs.logits # [K, seq_len, vocab]
                
                valid_candidates = 0
                
                for k in range(self.config.k_candidates):
                    seq = outputs[k]
                    
                    with torch.no_grad():
                        score = self.scorer.get_gem_score(seq, all_logits[k], prompt_len)
                        gem_scores.append(score)
                    
                    
                    shift_logits = all_logits[k, :-1, :]
                    shift_labels = seq[1:]
                    
                    
                    resp_logits = shift_logits[prompt_len-1:]
                    resp_labels = shift_labels[prompt_len-1:]
                    
                    if len(resp_labels) == 0:
                        log_probs_sum.append(torch.tensor(0.0).to(self.device))
                        continue
                        
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    token_loss = loss_fct(resp_logits, resp_labels) # -log(p)
                    token_log_probs = -token_loss
                    log_probs_sum.append(token_log_probs.sum())
                    valid_candidates += 1

                if valid_candidates < 2: 
                    continue

                
                scores_t = torch.stack(gem_scores)
                baseline = scores_t.mean()
                advantages = scores_t - baseline # Group Mean Centered
                
                
                if advantages.std() > 1e-6:
                    advantages = (advantages - advantages.mean()) / advantages.std()

                
                loss = 0
                for k in range(self.config.k_candidates):
                    loss += -1 * advantages[k].detach() * log_probs_sum[k]
                
                loss = loss / valid_candidates
                loss = loss / self.config.gradient_accumulation_steps
                
                loss.backward()
                accumulated_loss += loss.item()
                
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    pbar.set_postfix({"loss": accumulated_loss, "avg_score": baseline.item()})
                    accumulated_loss = 0

        print("=== GEM Optimization Completed ===")