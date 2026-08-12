#!/usr/bin/env python3
"""Channel-capacity bound: decode latents back to text with the AE's
RECONSTRUCT pathway, then answer with the base model over the reconstruction.

Phase recon (--phase recon, shardable): for each unique article among the
first --max-samples validation questions, greedily decode every chunk's 16
latents with the original AE checkpoint. Writes <out>/recon_<sha>.json.

Phase answer (--phase answer, single GPU, vLLM): ceiling-style prompt over the
concatenated reconstruction; accuracy on validation[0:max_samples].
"""

import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["recon", "answer"], required=True)
    ap.add_argument("--ae-checkpoint", default="data/autoencoder_pretrain_longctx_big/final")
    ap.add_argument("--latent-dir", default="data/quality_latents")
    ap.add_argument("--out", default="data/recon_answer")
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-recon-tokens", type=int, default=2048)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            open(os.path.join(args.latent_dir, "index_validation.jsonl"))]
    rows = rows[:args.max_samples]
    os.makedirs(args.out, exist_ok=True)

    if args.phase == "recon":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.ae_checkpoint)
        model = AutoModelForCausalLM.from_pretrained(
            args.ae_checkpoint, torch_dtype=torch.bfloat16).to(args.device)
        model.eval()
        embed = model.get_input_embeddings()
        shas = sorted({r["article_sha"] for r in rows})[args.shard_id::args.num_shards]
        prompt = ("<|im_start|>user\n" + "".join(f"<|L{i}|>" for i in range(16))
                  + "RECONSTRUCT<|im_end|>\n<|im_start|>assistant\n")
        pids = tok(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(args.device)
        li = {tok.convert_tokens_to_ids(f"<|L{i}|>") for i in range(16)}
        pos = [p for p, t in enumerate(pids[0].tolist()) if t in li]
        for sha in shas:
            lats = torch.load(os.path.join(args.latent_dir, "latents", f"{sha}.pt"),
                              map_location="cpu", weights_only=True)["latents"]
            chunks = []
            for c in range(lats.shape[0]):
                e = embed(pids)
                e[0, torch.tensor(pos, device=args.device), :] = \
                    lats[c].to(args.device, e.dtype)
                with torch.no_grad():
                    out = model.generate(
                        inputs_embeds=e, attention_mask=torch.ones_like(pids),
                        max_new_tokens=args.max_recon_tokens, do_sample=False,
                        repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
                chunks.append(tok.decode(out[0], skip_special_tokens=True))
            with open(os.path.join(args.out, f"recon_{sha}.json"), "w") as f:
                json.dump({"article_sha": sha, "chunks": chunks}, f)
            print(f"[recon shard {args.shard_id}] {sha} ({lats.shape[0]} chunks)", flush=True)

    else:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        import re
        recon = {}
        for p in glob.glob(os.path.join(args.out, "recon_*.json")):
            d = json.load(open(p))
            recon[d["article_sha"]] = "\n".join(d["chunks"])
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        prompts, metas = [], []
        for r in rows:
            o = r["options"]
            qb = (f"Question: {r['question']}\n"
                  f"A. {o[0]}\nB. {o[1]}\nC. {o[2]}\nD. {o[3]}\n\n"
                  "Reason step by step and output the final answer inside \\boxed{YOUR_FINAL_ANSWER}. "
                  "Your final answer must be one of A,B,C,D. Do not add any other contents inside the box.")
            user = ("The following document was compressed and then reconstructed; it may "
                    "contain minor corruptions. Read it and answer the question about the "
                    "original document — give your best supported answer.\n\n"
                    f"Document:\n{recon[r['article_sha']]}\n\n{qb}")
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            metas.append(r)
        llm = LLM(model="Qwen/Qwen3-4B", dtype="bfloat16", max_model_len=16384,
                  gpu_memory_utilization=0.9)
        sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=4096, seed=42)
        outs = llm.generate(prompts, sp)
        correct = 0
        for m, o in zip(metas, outs):
            mm = re.findall(r"\\boxed\{([^}]*)\}", o.outputs[0].text)
            pred = mm[-1].strip().strip("$").lower()[:1] if mm else ""
            correct += int(pred == "abcd"[m["answer"]])
        res = {"method": "recon_then_answer", "n": len(metas),
               "accuracy": correct / len(metas), "correct": correct}
        print(json.dumps(res))
        with open(os.path.join(args.out, "answer_result.json"), "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
