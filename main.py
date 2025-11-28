import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

from src.config import Config
from src.dataset import PreferenceDataset
from src.sft_trainer import run_sft
from src.gem_trainer import GEMTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/preference_data.jsonl")
    # This data is for example purposes; Embed your own data.
    args = parser.parse_args()
    
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    
    print(f"Loading {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.to(device)
    
    
    sft_dataset = PreferenceDataset(args.data_path, tokenizer, config, mode="sft")
    model = run_sft(model, sft_dataset, config, device)
    
    
    gem_dataset = PreferenceDataset(args.data_path, tokenizer, config, mode="gen")
    gem_trainer = GEMTrainer(model, tokenizer, config, device)
    
    gem_trainer.train_sega(gem_dataset)
    
    #Save
    print("Saving GEM-aligned model...")
    model.save_pretrained("output/gem_final")

if __name__ == "__main__":
    main()