# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import List, Optional

class CRF(nn.Module):
    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.LongTensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
    ) -> torch.Tensor:
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)  # 改为bool型默认mask

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # 拆分：bool_mask用于条件判断，float_mask用于数值计算
        bool_mask = mask.bool()
        float_mask = bool_mask.float()

        numerator = self._compute_score(emissions, tags, float_mask)
        denominator = self._compute_normalizer(emissions, bool_mask, float_mask)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        elif reduction == 'sum':
            return llh.sum()
        else:
            return llh.mean()

    def decode(self, emissions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> List[List[int]]:
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.bool)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        bool_mask = mask.bool()
        return self._viterbi_decode(emissions, bool_mask)

    def _validate(
        self,
        emissions: torch.Tensor,
        tags: Optional[torch.LongTensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(
        self, emissions: torch.Tensor, tags: torch.LongTensor, mask: torch.Tensor
    ) -> torch.Tensor:
        seq_length, batch_size = tags.shape

        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            score += self.transitions[tags[i-1], tags[i]] * mask[i]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
        self, emissions: torch.Tensor, bool_mask: torch.Tensor, float_mask: torch.Tensor
    ) -> torch.Tensor:
        seq_length = emissions.size(0)

        alpha = self.start_transitions + emissions[0]
        for i in range(1, seq_length):
            broadcast_alpha = alpha.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            next_alpha = broadcast_alpha + self.transitions + broadcast_emissions
            next_alpha = torch.logsumexp(next_alpha, dim=1)
            # 关键修复：用bool_mask做条件判断（torch.where要求bool）
            alpha = torch.where(bool_mask[i].unsqueeze(1), next_alpha, alpha)

        alpha += self.end_transitions
        return torch.logsumexp(alpha, dim=1)

    def _viterbi_decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        seq_length, batch_size = mask.shape

        backpointers = []
        alpha = self.start_transitions + emissions[0]
        for i in range(1, seq_length):
            broadcast_alpha = alpha.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            next_alpha = broadcast_alpha + self.transitions + broadcast_emissions

            next_alpha, max_idx = next_alpha.max(dim=1)
            backpointers.append(max_idx)
            alpha = torch.where(mask[i].unsqueeze(1), next_alpha, alpha)

        alpha += self.end_transitions
        max_alpha, max_idx = alpha.max(dim=1)
        best_tags = []
        backpointers = torch.stack(backpointers)

        for idx in range(batch_size):
            best_tag = [max_idx[idx].item()]
            seq_end = mask[:, idx].sum().item() - 1
            for bp in reversed(backpointers[:seq_end]):
                best_tag.append(bp[idx][best_tag[-1]].item())
            best_tag.reverse()
            best_tags.append(best_tag)

        # 关键修复：把 best_tagsS 改为 best_tags
        return best_tags