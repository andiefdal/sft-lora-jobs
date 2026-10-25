# sft-lora-jobs

QLoRA supervised fine-tuning runs for 30B-class open-weight models, executed as
detached jobs on [Modal](https://modal.com).

Personal research repo: the training loop, the Modal entrypoint that wraps it, and
the config I actually run. No serving layer, no public endpoints.

## What runs

| | |
|---|---|
| GPU | 1 × `RTX-PRO-6000` (96 GB) — enough for 4-bit 32B + LoRA |
| Job shape | detached, 24 h ceiling, one container at a time |
| Base model | Qwen2.5-32B-Instruct, frozen, loaded in NF4 |
| Adapters | LoRA r=32, alpha=64, target all linear projections |
| Data | my own support-chat and technical-doc corpora, mounted read-only at job start |
| Checkpoints | every 15 optimizer steps → Modal Volume `/output` |
| Eval | every 50 steps: validation loss + perplexity |
| Logging | one line per step: loss, lr, GPU%, tokens/s, grad-norm, mem% |

Everything lives for the duration of the container. The only things that outlive a
job are the LoRA adapters and checkpoint files on the Volume; the corpus is fetched
at start and disappears with the container.

## Why this shape

A 32B run does not fit in one sitting on a laptop, and renting a box by the month
for 6–8 hours of work a week is wasted money. Detached jobs mean I can start a run,
close the laptop, and come back to a finished checkpoint; the Volume means a
preempted container resumes from step N instead of step 0; per-second billing means
the idle days cost nothing.

## Log format

The trainer prints one line per optimizer step. This is the exact format (it is what
I grep when a run looks off):

```
[TRAIN] Step 000001 | Loss: 0.5842 | LR: 1.63e-04 | GPU: 96% | Tokens/s: 2144 | GradNorm: 0.812 | Mem: 88%
[INFO] Checkpoint saved: /output/checkpoint-15.pt (7.4 GB)
[EVAL] Validation loss: 0.3417 | Perplexity: 1.41
```

`logs/sample-run.log` is a trimmed excerpt: real step numbers, gaps between the
blocks are omitted runs/resumes, not edits to the values.

## Layout

```
train.py                       QLoRA loop (transformers + peft + bitsandbytes)
deploy_modal.py                Modal app: image, GPU, timeout, Volume mount
configs/rtxpro6000-1x_qlora.yaml   the config used for these runs
logs/sample-run.log            trimmed excerpt of an actual run
requirements.txt
```

## Usage

```bash
modal run --detach deploy_modal.py --config configs/rtxpro6000-1x_qlora.yaml
modal volume get finetune-output /checkpoint-latest ./ckpt/
```

Resume is automatic: if `/output/checkpoint-*` exists the trainer picks the highest
step and continues from there.

## Notes / limits

- 4-bit frozen base + bf16 LoRA is the only combination I got stable at 32B on one
  96 GB card; full fine-tuning is out of scope here.
- Loss spikes around step ~1.4k on this corpus are expected — I keep an eye on
  grad-norm, not on raw loss.
- Eval is a 2% held-out split, not a benchmark suite. It tells me whether a run is
  diverging, nothing more.
