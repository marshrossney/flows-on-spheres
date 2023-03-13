from itertools import chain
import math
import random

import torch

Tensor = torch.Tensor

PI = math.pi


def simple_fnn_conditioner(
    in_features: int,
    out_features: int,
    hidden_shape: list[int],
    activation: str = "Tanh",
    bias: bool = True,
) -> torch.nn.Sequential:
    layers = [
        torch.nn.Linear(f_in, f_out, bias=bias)
        for f_in, f_out in zip(
            [in_features, *hidden_shape], [*hidden_shape, out_features]
        )
    ]

    activation = getattr(torch.nn, activation)
    activations = [activation() for _ in hidden_shape] + [torch.nn.Identity()]

    net = torch.nn.Sequential(*list(chain(*zip(layers, activations))))

    def forward_pre_hook(module, inputs) -> Tensor:
        if isinstance(inputs, Tensor):
            return inputs.flatten(start_dim=1)
        else:
            return torch.cat(
                [x.flatten(start_dim=1) for x in inputs],
                dim=1,
            )

    net.register_forward_pre_hook(forward_pre_hook)

    return net


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
        elif random.random() < min(1, math.exp(proposal - current)):
            current = proposal
            idx += 1

        indices.append(idx)

    transitions = set(indices)
    transitions.discard(0)  # there was no transition *to* 0th state

    return len(transitions) / len(log_weights)


def effective_sample_size(log_weights: Tensor) -> float:
    ess = torch.exp(log_weights.logsumexp(0).mul(2) - log_weights.mul(2).logsumexp(0))
    ess.div_(len(log_weights))
    return float(ess)


def spherical_mesh(n: int) -> Tensor:
    theta = torch.linspace(0, PI, n)
    # theta = torch.acos(1 - 2 * torch.linspace(0, 1, n))
    phi = torch.linspace(0, 2 * PI, n)

    theta, phi = torch.meshgrid(theta, phi)

    x = theta.sin() * phi.cos()
    y = theta.sin() * phi.sin()
    z = theta.cos()

    return x, y, z


def batched_dot(x: Tensor, y: Tensor) -> Tensor:
    return (x * y).sum(dim=-1)


def batched_outer(x: Tensor, y: Tensor) -> Tensor:
    return x.unsqueeze(dim=-2) * y.unsqueeze(dim=-1)


def batched_mv(M: Tensor, v: Tensor) -> Tensor:
    return (M * v.unsqueeze(dim=-2)).sum(dim=-1)

# M = torch.rand(3, 3)
# v = torch.rand(3)
# res = batched_mv(M.view(1, 3, 3), v.view(1, 3))
# assert torch.allclose(M @ v, res)