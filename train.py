"""QLoRA SFT loop — prints one line per optimizer step in the format I grep.

Local smoke test:
    python train.py --config configs/rtxpro6000-1x_qlora.yaml --max-steps 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    return ap.parse_args()


def load_config(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg.setdefault("max_steps", 4000)
    cfg.setdefault("save_steps", 15)
    cfg.setdefault("eval_steps", 50)
    cfg.setdefault("output_dir", "/output")
    return cfg


def latest_checkpoint(out_dir: str) -> str | None:
    ckpts = sorted(Path(out_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(ckpts[-1]) if ckpts else None


def build_model(cfg: dict):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=quant, device_map={"": 0},
        attn_implementation="flash_attention_2",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"], bias="none", task_type="CAUSAL_LM",
        target_modules=cfg["lora"]["target_modules"],
    )
    return get_peft_model(model, lora), tok


def load_corpus(cfg: dict) -> list[dict]:
    src = Path(cfg["data_path"])
    if not src.exists():
        raise SystemExit(f"corpus not found at {src} — it is mounted at job start")
    rows = []
    for f in sorted(src.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    random.Random(cfg.get("seed", 0)).shuffle(rows)
    return rows


def batch(rows: list[dict], size: int, tok, max_len: int):
    for i in range(0, len(rows), size):
        chunk = rows[i:i + size]
        enc = tok([r["text"] for r in chunk], return_tensors="pt", truncation=True,
                  max_length=max_len, padding="max_length")
        enc["labels"] = enc["input_ids"].clone()
        yield enc


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    max_steps = args.max_steps or cfg["max_steps"]

    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    model, tok = build_model(cfg)
    rows = load_corpus(cfg)
    print(f"[INFO] trainer started on 1x RTX PRO 6000, output={out_dir}", flush=True)
    print(f"[INFO] {len(rows)} examples, {cfg['base_model']}, lora r={cfg['lora']['r']}", flush=True)

    resume_from = latest_checkpoint(out_dir)
    step = 0
    if resume_from:
        state = torch.load(os.path.join(resume_from, "trainer_state.pt"), map_location="cpu")
        model.load_state_dict(state["lora"], strict=False)
        step = int(state["step"])
        print(f"[INFO] resumed from {resume_from}", flush=True)

    batches = batch(rows, cfg["batch_size"], tok, cfg["max_seq_len"])
    opt = AdamW([p for p in model.parameters() if p.requires_grad],
                lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = get_cosine_schedule_with_warmup(opt, cfg["warmup_steps"], max_steps)

    model.train()
    t_last = time.time()
    tokens_window = 0
    while step < max_steps:
        b = next(batches, None)
        if b is None:
            batches = batch(rows, cfg["batch_size"], tok, cfg["max_seq_len"])
            b = next(batches)
        b = {k: v.to("cuda") for k, v in b.items()}
        out = model(**b)
        loss = out.loss
        (loss / cfg["grad_accum"]).backward()
        step += 1
        tokens_window += int(b["input_ids"].numel())

        if step % cfg["grad_accum"] == 0:
            gn = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            dt = max(time.time() - t_last, 1e-6)
            print(
                f"[TRAIN] Step {step:06d} | Loss: {loss.item():.4f} | "
                f"LR: {sched.get_last_lr()[0]:.2e} | GPU: {torch.cuda.utilization()}% | "
                f"Tokens/s: {int(tokens_window / dt)} | GradNorm: {float(gn):.3f} | "
                f"Mem: {int(torch.cuda.memory_reserved() / torch.cuda.get_device_properties(0).total_memory * 100)}%",
                flush=True,
            )
            t_last, tokens_window = time.time(), 0

        if step % cfg["save_steps"] == 0:
            path = os.path.join(out_dir, f"checkpoint-{step}.pt")
            torch.save({"lora": model.state_dict(), "step": step, "cfg": cfg}, path)
            size_gb = os.path.getsize(path) / 1e9
            print(f"[INFO] Checkpoint saved: {path} ({size_gb:.1f} GB)", flush=True)

        if step % cfg["eval_steps"] == 0:
            val = eval_loss(model, rows[:256], tok, cfg)
            print(f"[EVAL] Validation loss: {val:.4f} | Perplexity: {math.exp(val):.2f}", flush=True)


def eval_loss(model, rows, tok, cfg) -> float:
    import torch

    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for b in batch(rows, cfg["batch_size"], tok, cfg["max_seq_len"]):
            b = {k: v.to("cuda") for k, v in b.items()}
            total += float(model(**b).loss)
            n += 1
    model.train()
    return total / max(n, 1)


if __name__ == "__main__":
    main()
