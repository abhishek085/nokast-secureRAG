"""
Export the fine-tuned student as a standalone HuggingFace model.

Folds the LoRA adapter into the base Qwen2.5-0.5B weights (merge_and_unload) and
saves a self-contained model + tokenizer that loads with a single
`AutoModelForCausalLM.from_pretrained(...)` and serves directly under
transformers / vLLM. Runs inside the vllm container (CUDA torch + peft).

    bash scripts/merge.sh
"""
import os
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("STUDENT_MODEL", "Qwen/Qwen2.5-0.5B"))
    ap.add_argument("--adapter", default="models/adapters")
    ap.add_argument("--out", default="models/merged")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"base={args.base}  adapter={args.adapter}  ->  {args.out}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()  # bake the LoRA delta into the base weights

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)

    # ship the adapter's tokenizer (carries the chat template used in training)
    tok = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    tok.save_pretrained(args.out)

    print(f"merged model saved -> {args.out}")
    for f in sorted(os.listdir(args.out)):
        sz = os.path.getsize(os.path.join(args.out, f))
        print(f"  {f}  ({sz/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
