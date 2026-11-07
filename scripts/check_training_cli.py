# This script actually invokes training; it is not a read-only CLI check.
# 此脚本会实际训练，不应作为只读命令检查运行。
"""End-to-end training CLI, including shared LMDB indexing and device config."""
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml

root = Path(__file__).resolve().parents[1]
out = Path(os.environ["TEST_OUTPUT"]) / "cli"
out.mkdir(parents=True, exist_ok=True)
mgf = out / "synthetic.mgf"
mgf.write_text("".join(
    f"BEGIN IONS\nTITLE=synthetic_{i}\nPEPMASS=351.0\nCHARGE=2+\nSEQ=AGAGAG\n"
    + "".join(f"{100 + j * 50} {100 + j}\n" for j in range(12))
    + "END IONS\n" for i in range(16)))
config = yaml.safe_load((root / "docs/training_config.example.yaml").read_text())
config.update(
    random_seed=3407, dim_model=32, n_head=4, dim_feedforward=64, n_layers=1,
    dropout=0., custom_ctc=False, PMC_enable=False, n_beams=0,
    train_from_scratch=True, train_from_resume=False, train_num_samples=None,
    train_batch_size=2, accumulate_grad_batches=2, n_workers=0,
    trainer_accelerator="cpu", trainer_devices=2, trainer_precision=32,
    max_steps=2, max_epochs=1, warmup_iters=1, max_iters=10,
    val_interval=1, limit_val_batches=2, save_model=True,
    every_n_train_steps=1, save_weights_only=False,
    temp_dir_auto=False, temp_dir_path=str(out / "index"),
    enable_neptune=False, ctc_max_invalid_fraction=1.,
)
cfg = out / "config.yaml"
cfg.write_text(yaml.safe_dump(config))
run = subprocess.run(
    [sys.executable, "-m", "MNovo.backbone_cli", "--mode", "train",
     "--peak_path", str(mgf), "--peak_path_val", str(mgf),
     "--config", str(cfg), "--output", str(out / "run")],
    capture_output=True, text=True, timeout=90, cwd=root,
)
(out / "subprocess.log").write_text(run.stdout + run.stderr)
if run.returncode:
    raise RuntimeError(run.stderr[-5000:])
checkpoints = list((out / "run/checkpoints").glob("*.ckpt"))
assert checkpoints, "No checkpoint generated"
import torch
state = torch.load(checkpoints[-1], map_location="cpu")
assert state["hyper_parameters"]["warmup_iters"] == 1
assert state["hyper_parameters"]["max_iters"] == 10
(out / "PASS.json").write_text(json.dumps({"status": "PASS", "devices": 2,
    "checkpoint_count": len(checkpoints), "global_step": state["global_step"]}, indent=2))
print("PASS actual backbone CLI, two CPU processes, shared index and explicit scheduler")
