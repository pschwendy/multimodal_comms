#!/usr/bin/env python
"""Merge a LoRA AE adapter checkpoint onto its base into a plain, fully-loadable
AutoModelForCausalLM -- the contract pretrain_packed --init-from and the eval
scripts assume. LoRA AE checkpoints are adapter-only by design (merging inline
at save time would OOM the card at 8B); this is the explicit bridge.

    python training/programs/merge_adapter.py <base_model> <adapter_dir> <out_dir>

Runs on CPU by default (no GPU contention with training); pass a CUDA device as
the 4th arg to merge on GPU.
"""
import sys, torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from training.programs.pretrain_autoencoder import build_tokenizer
import json, os

base_name, adapter_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
device = sys.argv[4] if len(sys.argv) > 4 else "cpu"
# num_latents from the adapter's ae_config so the tokenizer adds the same <|Li|>.
K = json.load(open(os.path.join(adapter_dir, "ae_config.json")))["num_latents"]

tok = build_tokenizer(K, base_name)
dtype = torch.bfloat16 if device != "cpu" else torch.float32
base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype).to(device)
if base.get_input_embeddings().weight.shape[0] != len(tok):
    base.resize_token_embeddings(len(tok))
model = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
model.save_pretrained(out_dir)
tok.save_pretrained(out_dir)
with open(os.path.join(out_dir, "ae_config.json"), "w") as f:
    json.dump({"num_latents": K, "bottleneck_dim": None}, f)
print(f"merged {adapter_dir} onto {base_name} -> {out_dir}  (H={model.config.hidden_size}, vocab={len(tok)})")
