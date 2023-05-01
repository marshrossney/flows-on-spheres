from math import exp
from random import random
from typing import TypeAlias

import scipy
import torch

Tensor: TypeAlias = torch.Tensor


def metropolis_acceptance(log_weights: Tensor) -> float:
    log_weights = log_weights.tolist()
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


def effective_sample_size(log_weights: Tensor) -> float:
    ess = torch.exp(
        log_weights.logsumexp(0).mul(2) - log_weights.mul(2).logsumexp(0)
    )
    ess.div_(len(log_weights))
    return float(ess)


def integrated_autocorrelation(values: Tensor) -> Tensor:
    assert values.dim() == 1
    N = len(values)
    n = N // 2

    values = (values - values.mean()).numpy()
    corr = torch.from_numpy(
        scipy.signal.correlate(values, values, mode="same")
    )
    corr = corr[n:] / corr[n]  # normalise and take +ve shifts
    integrated = corr.cumsum(dim=0) - 0.5

    # Find optimal window
    tau_exp = (
        (2.0 / ((2 * integrated + 1) / (2 * integrated - 1)).log())
        .nan_to_num()
        .clamp(min=1e-6)
    )
    window = torch.arange(1, n + 1)
    g = (-window / tau_exp).exp() - (tau_exp / (window * N).sqrt())
    idx = torch.argmax((g[1:] < 0).int(), axis=-1)

    tau_int = integrated[idx]

    return float(tau_int)
