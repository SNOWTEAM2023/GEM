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
        
        self.optimizer = AdamW(self.model.parameters(), lr=1e-6) 

    def train_sega(self, dataset):
        print("=== Starting Step 2: GEM Optimization (SEGA) ===")
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        
        self.model.train()
        
        for epoch in range(self.config.sega_epochs):
            pbar = tqdm(dataloader, desc=f"SEGA Epoch {epoch+1}")
            accumulated_loss = 0
            
            for step, prompt_batch in enumerate(pbar):
                prompt_str = prompt_batch[0]
                
         
                self.model.eval() 
                
                inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.device)
                prompt_len = inputs.input_ids.shape[1]
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=True,
                        temperature=self.config.temperature,
                        num_return_sequences=self.config.k_candidates,
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=True 
                    )
                
                
                self.model.train()
                
                
                
                gem_scores = []
                log_probs_sum = []
                valid_candidates = 0
                
                
                
                for k in range(self.config.k_candidates):
                    seq = outputs[k:k+1] # [1, seq_len]
                    
                    
                    with torch.no_grad():
                        self.model.eval() 
                        out_eval = self.model(seq)
                        score = self.scorer.get_gem_score(seq[0], out_eval.logits[0], prompt_len)
                        gem_scores.append(score)
                        self.model.train() # 切回 train
                    
                    
                    out_train = self.model(seq)
                    all_logits = out_train.logits
                    
                    
                    logits_seg = all_logits[0, prompt_len-1:-1, :]
                    labels_seg = seq[0, prompt_len:]
                    
                    if len(labels_seg) == 0:
                        log_probs_sum.append(torch.tensor(0.0).to(self.device))
                        continue
                        
                    
                    ce = torch.nn.functional.cross_entropy(logits_seg, labels_seg, reduction='none')
                    token_log_probs = -ce
                    log_probs_sum.append(token_log_probs.sum())
                    valid_candidates += 1

                if valid_candidates < 2: 
                    continue

                
                scores_t = torch.stack(gem_scores)
                baseline = scores_t.mean()
                advantages = scores_t - baseline
                

                if advantages.std() > 1e-6:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Loss = - sum( Advantage * log_prob )
                loss = 0
                for k in range(self.config.k_candidates):
               
                    if isinstance(log_probs_sum[k], torch.Tensor) and log_probs_sum[k].requires_grad:
                        loss += -1 * advantages[k].detach() * log_probs_sum[k]
                
                if isinstance(loss, torch.Tensor):
                    loss = loss / valid_candidates
                    loss.backward()
                    accumulated_loss += loss.item()
                

                self.optimizer.step()
                self.optimizer.zero_grad()
                pbar.set_postfix({"loss": accumulated_loss, "avg_score": baseline.item()})
                accumulated_loss = 0

        print("=== GEM Optimization Completed ===")