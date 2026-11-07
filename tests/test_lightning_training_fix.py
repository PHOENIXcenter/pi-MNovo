# Training integration fixture; running it can create checkpoints.
# 训练集成测试可能写入检查点，不属于纯推理验证。
"""Exercise actual Spec2Pep training hooks on a tiny CPU transformer."""
import os
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from MNovo.denovo.model import Spec2Pep


class Spectra(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        peaks = torch.stack((torch.linspace(100, 900, 12),
                             torch.rand(12, generator=generator)), dim=-1)
        case = os.environ.get("TEST_CASE", "valid")
        invalid = case == "all_empty" or (case == "empty_rank" and index % 2 == 1)
        return peaks, torch.tensor([700., 2., 351.]), "A" * 80 if invalid else "AGAG"


if __name__ == "__main__":
    torch.set_num_threads(1)
    pl.seed_everything(3407, workers=True)
    devices = int(sys.argv[1])
    output = Path(sys.argv[2])
    model = Spec2Pep(
        custom_ctc_loss=False, dim_model=32, n_head=4,
        dim_feedforward=64, n_layers=1, max_length=40, dropout=0.,
        n_beams=0, PMC_enable=False, warmup_iters=1, max_iters=10,
        lr=1e-4, weight_decay=8e-5, ctc_max_invalid_fraction=1.,
    )
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    trainer = pl.Trainer(
        accelerator="cpu", devices=devices,
        strategy=DDPStrategy(process_group_backend="gloo", find_unused_parameters=True)
        if devices > 1 else None,
        max_steps=4 if len(sys.argv) > 3 else 2, max_epochs=10,
        accumulate_grad_batches=2, replace_sampler_ddp=False,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        num_sanity_val_steps=0, deterministic=True,
    )
    dataset = Spectra()
    sampler = DistributedSampler(dataset, num_replicas=devices,
                                 rank=int(os.environ.get("LOCAL_RANK", 0)), shuffle=False)
    loader = DataLoader(dataset, batch_size=4 // devices, sampler=sampler)
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader,
                ckpt_path=sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "fresh" else None)
    trainer.save_checkpoint(str(output))
    assert model._ctc_counts["train"]["seen"] == (32 if len(sys.argv) > 3 else 16)
    case = os.environ.get("TEST_CASE", "valid")
    expected_excluded = {"valid": 0, "empty_rank": 8, "all_empty": 16}[case]
    assert model._ctc_counts["train"]["excluded"] == expected_excluded
    expected_updates = 0 if case == "all_empty" else (4 if len(sys.argv) > 3 else 2)
    assert trainer.lr_scheduler_configs[0].scheduler.last_epoch == expected_updates
    if case == "all_empty":
        assert not trainer.optimizers[0].state
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, initial[key], atol=0, rtol=0)
    if trainer.is_global_zero:
        print(f"PASS actual Spec2Pep Lightning CPU devices={devices}, accumulation=2 case={case}")
