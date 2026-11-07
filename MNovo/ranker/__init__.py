# Candidate ranking and expert routing operate on an existing candidate pool.
# 专家排序与路由都在已生成的候选池内进行。
"""Candidate ranking, expert evidence, and observable routing."""

from .model import ResidualCandidateRanker

__all__ = ["ResidualCandidateRanker"]
