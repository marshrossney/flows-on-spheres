from math import exp
from random import random
from typing import TypeAlias

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
