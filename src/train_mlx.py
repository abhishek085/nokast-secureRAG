"""
MLX Training Module for Nokast-secureRAG
Fine-tunes Qwen2.5-0.5B using mlx_lm with LoRA (rank 8)
"""
import json
import os
from pathlib import Path
from typing import Dict, List

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate
    from mlx_lm.tuner.trainer import train
    from mlx_lm.tuner.utils import build_schedule
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("Warning: MLX not available. Training functionality limited.")


class QwenPromptInjectionTrainer:
    """Fine-tune Qwen2.5-0.5B for prompt injection detection"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        data_path: str = "data/rag_samples.jsonl",
        output_dir: str = "models/adapters",
        lora_rank: int = 8,
    ):
        self.model_name = model_name
        self.data_path = data_path
        self.output_dir = output_dir
        self.lora_rank = lora_rank
        
    def prepare_data(self, train_split: float = 0.9) -> tuple:
        """Prepare training and validation data"""
        print(f"Loading data from {self.data_path}...")
        
        samples = []
        with open(self.data_path, 'r') as f:
            for line in f:
                samples.append(json.loads(line))
        
        # Shuffle
        import random
        random.shuffle(samples)
        
        # Split into train/val
        split_idx = int(len(samples) * train_split)
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]
        
        print(f"Train samples: {len(train_samples)}")
        print(f"Validation samples: {len(val_samples)}")
        
        return train_samples, val_samples
    
    def format_prompt(self, query: str, label: str = None, reasoning: str = None) -> str:
        """Format sample as instruction-following prompt"""
        instruction = "Analyze the following query for prompt injection attempts. Classify it as SAFE or MALICIOUS and provide reasoning."
        
        if label and reasoning:
            # Training format
            return f"""<|im_start|>system
You are a security expert specializing in prompt injection detection.<|im_end|>
<|im_start|>user
{instruction}

Query: {query}<|im_end|>
<|im_start|>assistant
Label: {label}
Reasoning: {reasoning}<|im_end|>"""
        else:
            # Inference format
            return f"""<|im_start|>system
You are a security expert specializing in prompt injection detection.<|im_end|>
<|im_start|>user
{instruction}

Query: {query}<|im_end|>
<|im_start|>assistant
"""
    
    def save_formatted_data(self, samples: List[Dict], output_path: str):
        """Save formatted data for MLX training"""
        formatted_samples = []
        
        for sample in samples:
            text = self.format_prompt(
                query=sample["query"],
                label=sample["label"],
                reasoning=sample["reasoning"]
            )
            formatted_samples.append({"text": text})
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(formatted_samples, f, indent=2)
        
        print(f"Saved formatted data to {output_path}")
    
    def train_model(
        self,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        warmup_steps: int = 100,
    ):
        """Train the model using MLX LoRA"""
        print(f"Starting training with LoRA rank {self.lora_rank}...")
        
        # Prepare data
        train_samples, val_samples = self.prepare_data()
        
        # Save formatted data
        train_path = "data/train_formatted.json"
        val_path = "data/val_formatted.json"
        
        self.save_formatted_data(train_samples, train_path)
        self.save_formatted_data(val_samples, val_path)
        
        # Training configuration
        config = {
            "model": self.model_name,
            "train": True,
            "data": train_path,
            "valid_data": val_path,
            "adapter_file": os.path.join(self.output_dir, "adapters.npz"),
            "iters": num_epochs * (len(train_samples) // batch_size),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "steps_per_report": 10,
            "steps_per_eval": 100,
            "save_every": 100,
            "test": False,
            "test_batches": 100,
            "lora_layers": 16,
            "lora_rank": self.lora_rank,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
        }
        
        print("\nTraining Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("TRAINING STARTED")
        print("="*60)
        
        # Note: This is a simplified training setup
        # In production, you would call mlx_lm training functions
        print("""
To train the model, run:
    python -m mlx_lm.lora \\
        --model {model} \\
        --train \\
        --data {train_data} \\
        --valid-data {val_data} \\
        --batch-size {batch_size} \\
        --lora-layers {lora_layers} \\
        --lora-rank {lora_rank} \\
        --iters {iters} \\
        --learning-rate {learning_rate} \\
        --adapter-file {adapter_file}
        
Or use the mlx_lm API directly in your code.
""".format(**config))
        
        # Save config for reference
        config_path = os.path.join(self.output_dir, "train_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nTraining configuration saved to {config_path}")
        print(f"Adapter will be saved to {config['adapter_file']}")
        
        return config


def main():
    """Main entry point"""
    trainer = QwenPromptInjectionTrainer(
        model_name="Qwen/Qwen2.5-0.5B",
        data_path="data/rag_samples.jsonl",
        output_dir="models/adapters",
        lora_rank=8
    )
    
    config = trainer.train_model(
        num_epochs=3,
        batch_size=4,
        learning_rate=1e-4,
    )


if __name__ == "__main__":
    main()
