"""Modal entrypoint for the QLoRA runs.

    modal run --detach deploy_modal.py --config configs/rtxpro6000-1x_qlora.yaml
"""

from pathlib import Path

import modal

APP = "sft-lora-jobs"

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel", add_python="3.11")
    .apt_install("git", "libgomp1")
    .pip_install(
        "transformers==4.45.2",
        "peft==0.13.2",
        "bitsandbytes==0.44.1",
        "accelerate==1.0.1",
        "datasets==3.0.1",
        "flash-attn==2.7.0.post2",
        "pyyaml==6.0.2",
    )
    .env({"HF_HOME": "/output/hf", "TOKENIZERS_PARALLELISM": "false"})
)

app = modal.App(APP)

output = modal.Volume.from_name("finetune-output", create_if_missing=True)

# read-only corpus mount; path inside the container is what train.py reads
corpus = modal.Volume.from_name("sft-corpus", create_if_missing=True)


@app.function(
    image=image,
    gpu="RTX-PRO-6000:1",
    cpu=2.0,
    memory=32768,
    timeout=24 * 3600,
    volumes={"/output": output, "/data": corpus},
    secrets=[modal.Secret.from_name("hf-token", required=False)],
)
def train(config: str = "configs/rtxpro6000-1x_qlora.yaml") -> str:
    import subprocess

    cmd = ["python", "-u", "train.py", "--config", config]
    proc = subprocess.run(cmd, check=False)
    output.commit()
    return f"exit={proc.returncode}"


@app.local_entrypoint()
def main(config: str = "configs/rtxpro6000-1x_qlora.yaml"):
    print(train.remote(config))
