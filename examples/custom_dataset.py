"""
Example: Generating Custom Training Data
Shows how to create custom datasets for specific use cases
"""
import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_gen import RAGDataGenerator


def create_domain_specific_dataset():
    """Create a dataset for a specific domain (e.g., healthcare)"""
    
    # Customize for your domain
    generator = RAGDataGenerator(
        model_name="kimi-k2.5:cloud",
        num_samples=100  # Smaller dataset for demo
    )
    
    # Override templates for healthcare domain
    generator.safe_templates = [
        "What are the symptoms of {topic}?",
        "How is {topic} diagnosed?",
        "What are treatment options for {topic}?",
        "Explain {topic} to a patient.",
    ]
    
    generator.topics = [
        "diabetes", "hypertension", "asthma",
        "arthritis", "migraine", "pneumonia"
    ]
    
    print("\n" + "="*70)
    print("EXAMPLE: Custom Domain Dataset Generation")
    print("="*70 + "\n")
    
    # Generate samples
    samples = []
    
    print("Generating 5 safe healthcare queries...")
    for _ in range(5):
        sample = generator.generate_sample(is_malicious=False)
        samples.append(sample)
        print(f"  • {sample['query']}")
    
    print("\nGenerating 5 malicious queries...")
    for _ in range(5):
        sample = generator.generate_sample(is_malicious=True)
        samples.append(sample)
        print(f"  • {sample['query'][:60]}...")
    
    # Save custom dataset
    output_path = "data/healthcare_samples.jsonl"
    os.makedirs("data", exist_ok=True)
    
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"\n✅ Dataset saved to {output_path}")
    print(f"Total samples: {len(samples)}")


if __name__ == "__main__":
    create_domain_specific_dataset()
