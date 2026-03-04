# -*- coding: utf-8 -*-
"""\
线性链 CRF（纯 PyTorch 实现，无额外依赖）

接口设计参考 torchcrf：
- forward(emissions, tags, mask) -> log_likelihood
- decode(emissions, mask) -> List[List[int]]

emissions: (batch, seq_len, num_tags)  (batch_first=True)
mask:      (batch, seq_len) bool/byte, True 表示有效位置
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self, num_tags: int, batch_first: bool = True):
        super().__init__()
        self.num_tags = int(num_tags)
        self.batch_first = bool(batch_first)

        # 转移分数：从 i -> j
        self.transitions = nn.Parameter(torch.empty(self.num_tags, self.num_tags))
        # 起始/结束分数
        self.start_transitions = nn.Parameter(torch.empty(self.num_tags))
        self.end_transitions = nn.Parameter(torch.empty(self.num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """返回 log-likelihood（越大越好）。训练时通常用 loss = -log_likelihood"""
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)

        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)
        else:
            mask = mask.to(torch.bool)

        log_den = self._compute_log_partition_function(emissions, mask)
        log_num = self._compute_score(emissions, tags, mask)
        llh = log_num - log_den

        if reduction == "none":
            return llh
        if reduction == "sum":
            return llh.sum()
        if reduction == "mean":
            return llh.mean()
        raise ValueError(f"invalid reduction: {reduction}")

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> List[List[int]]:
        """Viterbi 解码"""
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)

        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
        else:
            mask = mask.to(torch.bool)

        return self._viterbi_decode(emissions, mask)

    # ---------------- internal ----------------

    def _compute_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """给定 gold tags 的路径分数"""
        batch_size, seq_len, num_tags = emissions.shape
        assert num_tags == self.num_tags

        # 第一个位置
        score = self.start_transitions[tags[:, 0]] + emissions[:, 0, :].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for t in range(1, seq_len):
            mask_t = mask[:, t]
            emit_t = emissions[:, t, :].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)
            trans_t = self.transitions[tags[:, t - 1], tags[:, t]]
            score = score + (emit_t + trans_t) * mask_t

        # 结束：找到每个序列最后一个有效位置的 tag
        last_tag_indices = mask.long().sum(dim=1) - 1  # (B,)
        last_tags = tags.gather(1, last_tag_indices.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]
        return score

    def _compute_log_partition_function(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """前向算法计算 logZ"""
        batch_size, seq_len, num_tags = emissions.shape
        assert num_tags == self.num_tags

        # alpha: (B, num_tags)
        alpha = self.start_transitions + emissions[:, 0]  # (B, C)

        for t in range(1, seq_len):
            emit_t = emissions[:, t].unsqueeze(2)  # (B, C, 1)
            # broadcast: (B, from, to)
            score_t = alpha.unsqueeze(2) + self.transitions.unsqueeze(0) + emit_t.transpose(1, 2)
            # logsumexp over from
            new_alpha = torch.logsumexp(score_t, dim=1)  # (B, to)

            mask_t = mask[:, t].unsqueeze(1)
            alpha = torch.where(mask_t, new_alpha, alpha)

        alpha = alpha + self.end_transitions
        return torch.logsumexp(alpha, dim=1)

    def _viterbi_decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        batch_size, seq_len, num_tags = emissions.shape

        # score: (B, C)
        score = self.start_transitions + emissions[:, 0]
        history = []  # list of (B, C) backpointers

        for t in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)  # (B, from, 1)
            broadcast_trans = self.transitions.unsqueeze(0)  # (B, from, to)
            next_score = broadcast_score + broadcast_trans
            # best previous tag for each current tag
            best_score, best_path = next_score.max(dim=1)  # (B, to)
            best_score = best_score + emissions[:, t]

            mask_t = mask[:, t].unsqueeze(1)
            score = torch.where(mask_t, best_score, score)
            history.append(best_path)

        score = score + self.end_transitions
        best_last_score, best_last_tag = score.max(dim=1)  # (B,)

        # 回溯
        best_paths: List[List[int]] = []
        for i in range(batch_size):
            seq_end = int(mask[i].long().sum().item())
            last_tag = int(best_last_tag[i].item())
            path = [last_tag]
            for backpointers in reversed(history[: seq_end - 1]):
                last_tag = int(backpointers[i, last_tag].item())
                path.append(last_tag)
            path.reverse()
            best_paths.append(path)
        return best_paths
