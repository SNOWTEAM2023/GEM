import json
from torch.utils.data import Dataset

class PreferenceDataset(Dataset):
    def __init__(self, data_path, tokenizer, config, mode="sft"):
        
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        self.tokenizer = tokenizer
        self.config = config
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt_text = item['prompt']
        
        
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": prompt_text}
        ]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        if self.mode == "sft":
            
            chosen_text = item['chosen']
            full_text = formatted_prompt + chosen_text + self.tokenizer.eos_token
            
            # Tokenize
            enc = self.tokenizer(
                full_text, 
                max_length=self.config.max_length, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            )
            input_ids = enc.input_ids.squeeze(0)
            labels = input_ids.clone()
            
            
            prompt_ids = self.tokenizer(formatted_prompt, return_tensors="pt").input_ids
            prompt_len = prompt_ids.shape[1]
            if prompt_len < len(labels):
                labels[:prompt_len] = -100
                
            return {"input_ids": input_ids, "labels": labels, "attention_mask": enc.attention_mask.squeeze(0)}
            
        elif self.mode == "gen":
            
            return formatted_prompt