"""Run the experiment on a cloud GPU with the full-size model.

    pip install modal && modal setup
    modal run modal_run.py                       # 8B model, 8 examples
    modal run modal_run.py --scaffold True        # add the recovery test
"""

from __future__ import annotations

import pathlib

import modal

DIR = pathlib.Path(__file__).resolve().parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "datasets>=3,<4", "accelerate",
                 "sentencepiece", "protobuf")
    .add_local_dir(DIR, remote_path="/root/exp")
)
app = modal.App("does-compression-change-behavior")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(image=image, gpu="A100-80GB", timeout=3600,
              volumes={"/root/.cache/huggingface": hf_cache})
def run(model: str, num_examples: int, samples: int, scaffold: bool):
    import subprocess
    cmd = ["python", "/root/exp/run_experiment.py", "--model", model,
           "--num-examples", str(num_examples), "--samples", str(samples),
           "--out", "/root/exp/results"]
    if scaffold:
        cmd.append("--scaffold")
    subprocess.run(cmd, check=True, cwd="/root/exp")
    hf_cache.commit()


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-8B", num_examples: int = 8,
         samples: int = 8, scaffold: bool = False):
    run.remote(model, num_examples, samples, scaffold)
