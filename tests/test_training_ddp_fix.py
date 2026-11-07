# Compare distributed valid-sample accounting across ranks.
# 核对不同进程之间有效样本的计数。
"""CPU/Gloo two-rank regression tests; no scientific inference or GPU use."""
import datetime
import json
import os
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from MNovo.training.training_ctc import masked_ctc_loss
from MNovo.training.distributed_training import distributed_ctc, check_finite_gradients


def sample(case, micro=0):
    generator = torch.Generator().manual_seed(50 + micro)
    x = torch.randn(7, 6, 4, generator=generator)
    labels = torch.tensor([[1, 2, 1]] * 6)
    lengths = torch.full((6,), 3)
    if case == "unequal":
        lengths[4:] = 0
    if case == "empty_rank":
        lengths[3:] = 0
    if case == "all_empty":
        lengths[:] = 0
    return x, labels, lengths


def reference(case, accumulation):
    torch.manual_seed(7)
    net = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.1)
    # Seed Adam momentum before the all-empty update.
    x, y, lengths = sample("valid")
    masked_ctc_loss(net(x), y, lengths, 0).loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    initial = [p.detach().clone() for p in net.parameters()]
    reports = []
    for micro in range(accumulation):
        x, y, lengths = sample(case, micro)
        result = masked_ctc_loss(net(x), y, lengths, 0)
        if result.loss is not None:
            reports.append(result.loss.detach())
            (result.loss / accumulation).backward()
        else:
            reports.append(torch.tensor(0.))
    gradients = [None if p.grad is None else p.grad.clone() for p in net.parameters()]
    opt.step()
    return initial, gradients, [p.detach().clone() for p in net.parameters()], reports


def worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=45))
    results = []
    for case, accumulation in [
        ("valid", 1), ("unequal", 1), ("empty_rank", 1),
        ("all_empty", 1), ("unequal", 3), ("empty_rank", 3)
    ]:
        # Reference calculation must not participate in distributed reductions.
        initial, expected_grad, expected_params, expected_loss = reference(case, accumulation)
        torch.manual_seed(7)
        net = DDP(torch.nn.Linear(4, 4))
        opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.1)
        x, y, lengths = sample("valid")
        sl = slice(rank * 3, (rank + 1) * 3)
        logits = net(x[:, sl])
        res = masked_ctc_loss(logits, y[sl], lengths[sl], 0)
        loss, _, _, _ = distributed_ctc(res, logits)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        for p, e in zip(net.module.parameters(), initial):
            torch.testing.assert_close(p, e, atol=2e-7, rtol=2e-6)
        has_data = False
        for micro in range(accumulation):
            x, y, lengths = sample(case, micro)
            logits = net(x[:, sl])
            res = masked_ctc_loss(logits, y[sl], lengths[sl], 0)
            loss, report, count, total = distributed_ctc(res, logits)
            assert total == 6
            torch.testing.assert_close(report, expected_loss[micro], atol=2e-6, rtol=2e-6)
            has_data |= count > 0
            (loss / accumulation).backward()
            check_finite_gradients(net.parameters())
        if not has_data:
            for p in net.parameters():
                p.grad = None
        for p, expected in zip(net.module.parameters(), expected_grad):
            if expected is None:
                assert p.grad is None
            else:
                torch.testing.assert_close(p.grad, expected, atol=2e-6, rtol=2e-5)
        opt.step()
        for p, expected in zip(net.module.parameters(), expected_params):
            torch.testing.assert_close(p, expected, atol=2e-7, rtol=2e-6)
        results.append({"case": case, "accumulation": accumulation, "status": "PASS"})
    # Only one rank has a bad gradient: both must reject the update.
    for p in net.parameters():
        p.grad = torch.ones_like(p)
    if rank == 1:
        next(net.parameters()).grad.flatten()[0] = float("nan")
    try:
        check_finite_gradients(net.parameters())
    except FloatingPointError:
        results.append({"case": "one_rank_nan_gradient", "status": "PASS"})
    else:
        raise AssertionError("Nonfinite gradient was not rejected on both ranks")
    Path(output, f"rank_{rank}.json").write_text(json.dumps(results, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    out = Path(os.environ["TEST_OUTPUT"])
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(worker, args=("file://" + directory + "/rendezvous", str(out)),
                 nprocs=2, join=True)
    print("PASS: two real CPU/Gloo DDP ranks; loss, gradients, AdamW updates and empty batches")
