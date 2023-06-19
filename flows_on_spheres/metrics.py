from functools import cached_property
from math import exp
from random import random
from typing import TypeAlias

import scipy
import torch

Tensor: TypeAlias = torch.Tensor


class LogWeightMetrics:
    def __init__(self, log_weights: Tensor):
        log_weights = log_weights.detach().squeeze()
        assert log_weights.dim() == 1
        self.log_weights = log_weights

    @cached_property
    def metropolis_acceptance(self) -> float:
        log_weights = self.log_weights.tolist()
        current = log_weights.pop(0)

        idx = 0
        indices = []

        for proposal in log_weights:
            # Deal with this case separately to avoid overflow
            if proposal > current:
                current = proposal
                idx += 1
            elif random() < min(1, exp(proposal - current)):
                current = proposal
                idx += 1

            indices.append(idx)

        transitions = set(indices)
        transitions.discard(0)  # there was no transition *to* 0th state

        return len(transitions) / len(log_weights)

    @property
    def effective_sample_size(self) -> float:
        ess = torch.exp(
            self.log_weights.logsumexp(0).mul(2)
            - self.log_weights.mul(2).logsumexp(0)
        )
        ess /= len(self.log_weights)
        return float(ess)

    @property
    def mean(self) -> float:
        return self.log_weights.mean().item()

    @property
    def variance(self) -> float:
        return self.log_weights.var().item()

    @property
    def min(self) -> float:
        return self.log_weights.min().item()

    @property
    def max(self) -> float:
        return self.log_weights.max().item()

    def as_dict(self) -> dict[str, float]:
        return {
            "metropolis_acceptance": self.metropolis_acceptance,
            "effective_sample_size": self.effective_sample_size,
            "mean": self.mean,
            "variance": self.variance,
            "min": self.min,
            "max": self.max,
        }


def integrated_autocorrelation(values: Tensor, factor: float = 2.0) -> Tensor:
    assert values.dim() == 2

    _, N = values.shape
    n = N // 2

    values = values - values.mean(dim=1, keepdim=True)

    autocorr = []
    for v in values:
        corr = torch.from_numpy(scipy.signal.correlate(v, v, mode="same"))
        corr = corr[n:] / corr[n]  # normalise and take +ve shifts
        autocorr.append(torch.as_tensor(corr))

    integrated = torch.stack(autocorr).cumsum(dim=1) - 0.5

    # Find optimal window
    tau_exp = (
        (factor / ((2 * integrated + 1) / (2 * integrated - 1)).log())
        .nan_to_num()
        .clamp(min=1e-6)
    )
    window = torch.arange(1, n + 1)
    g = (-window / tau_exp).exp() - (tau_exp / (window * N).sqrt())
    idx = torch.argmax((g[1:] < 0).int(), axis=1) + 1

    tau_int = integrated[:, idx]

    return tau_int
