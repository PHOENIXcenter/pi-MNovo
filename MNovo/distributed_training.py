# Weight CTC loss by valid-sample counts across ranks.
# 跨进程按有效样本数加权 CTC 损失。
"""Training-only distributed accounting; prediction code does not use this."""

import torch
import torch.distributed as dist


def distributed_ctc(result, logits):
    """DDP averages rank gradients, so scale local sums by world/global count."""
    local_count = int(result.valid.sum())
    counts = torch.tensor(
        [local_count, len(result.reasons)], dtype=torch.long, device=logits.device
    )
    distributed = dist.is_available() and dist.is_initialized()
    world = dist.get_world_size() if distributed else 1
    if distributed:
        dist.all_reduce(counts)
    count, total = counts.tolist()
    # An empty rank must still traverse the forward graph and join DDP backward.
    local_sum = (result.loss * local_count if result.loss is not None else
                 torch.nan_to_num(logits).mul(0).sum())
    reported = local_sum.detach().clone()
    if distributed:
        dist.all_reduce(reported)
    # The optimization loss accounts for DDP averaging; the reported mean does not.
    # 反传损失须抵消 DDP 的梯度平均，日志均值则直接除以全局有效数。
    return local_sum * (world / max(count, 1)), reported / max(count, 1), count, total


def check_finite_gradients(parameters):
    parameters = list(parameters)
    bad = torch.zeros((), dtype=torch.int32, device=parameters[0].device)
    for parameter in parameters:
        if parameter.grad is not None:
            bad.bitwise_or_((~torch.isfinite(parameter.grad).all()).to(bad.dtype))
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    if bad.item():
        raise FloatingPointError("Non-finite gradients on at least one rank")
