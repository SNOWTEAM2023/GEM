import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config import Config
from src.dataset import PreferenceDataset
from src.sft_trainer import run_sft
from src.gem_trainer import GEMTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/preference_data.jsonl")
    args = parser.parse_args()
    
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    print(f"Loading {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    dtype = torch.float16
    if torch.cuda.is_bf16_supported():
        print("BFloat16 supported. Using bfloat16 to prevent NaN.")
        dtype = torch.bfloat16
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        trust_remote_code=True
    )
    
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  
    
    model.config.use_cache = False 
    
    model.to(device)
    
    # Step 1: SFT
    sft_dataset = PreferenceDataset(args.data_path, tokenizer, config, mode="sft")
    model = run_sft(model, sft_dataset, config, device)
    
    # Step 2: GEM / SEGA
    gem_dataset = PreferenceDataset(args.data_path, tokenizer, config, mode="gen")
    gem_trainer = GEMTrainer(model, tokenizer, config, device)
    
    gem_trainer.train_sega(gem_dataset)
    
    print("Saving GEM-aligned model...")
    model.save_pretrained("output/gem_final", safe_serialization=True)
    tokenizer.save_pretrained("output/gem_final")

if __name__ == "__main__":
    main()