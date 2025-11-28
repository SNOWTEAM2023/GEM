from dataclasses import dataclass

@dataclass
class Config:
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # It features a simple implementation, is compatible with most environments, and is VRAM-friendly. You can switch to meta-llama/Llama-3.1-8B-Instruct.
    max_length: int = 512
    
    sft_lr: float = 2e-5
    sft_epochs: int = 1
    sft_batch_size: int = 4
    
    k_candidates: int = 4        
    temperature: float = 0.9     
    sega_lr: float = 1e-5
    sega_epochs: int = 1
    gradient_accumulation_steps: int = 2
    
    lambda_weight: float = 1.0   
    top_m_ratio: float = 0.1     
    
    system_prompt: str = (
        "You are a helpful assistant. "
        "Think step-by-step and then provide your Final Answer."
    )
    #Sample prompt – you can try your own prompt！