import torch
import torch.nn.functional as F

class EntropyScorer:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def compute_entropy(self, logits):
        
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        # 避免 nan
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy

    def get_gem_score(self, input_ids, logits, prompt_len):

        
        
        response_ids = input_ids[prompt_len:]
        
        response_logits = logits[prompt_len-1 : -1]
        
        if len(response_ids) == 0:
            return torch.tensor(0.0).to(logits.device)

        
        token_entropies = self.compute_entropy(response_logits) # [seq_len]

        
        full_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        sep = "Final Answer"
        
        split_idx = -1
        if sep in full_text:
            
            split_idx = int(len(token_entropies) * 0.75)
        else:
            split_idx = int(len(token_entropies) * 0.8)
            
        h_cot = token_entropies[:split_idx]
        h_final = token_entropies[split_idx:]
        
        
        score_final = torch.mean(h_final) if len(h_final) > 0 else torch.tensor(5.0).to(logits.device)
        
        
        score_cot = torch.tensor(0.0).to(logits.device)
        if len(h_cot) > 0:
            m = max(1, int(len(h_cot) * self.config.top_m_ratio))
            
            top_m_vals, _ = torch.topk(h_cot, m)
            score_cot = torch.mean(top_m_vals)
            
        
        gem_score = -score_final + (self.config.lambda_weight * score_cot)
        
        return gem_score