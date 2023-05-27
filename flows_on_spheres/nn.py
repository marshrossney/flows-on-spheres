from itertools import chain
from typing import Callable, Optional, TypeAlias

import torch
import torch.nn as nn

Tensor: TypeAlias = torch.Tensor


def make_fnn(
    size_out: int,
    hidden_shape: list[int],
    activation: Optional[str],
    size_in: Optional[int] = None,
):
    layers = [
        (
            torch.nn.Linear(f_in, f_out)
            if f_in is not None
            else nn.LazyLinear(f_out)
        )
        for f_in, f_out in zip(
            [size_in, *hidden_shape], [*hidden_shape, size_out]
        )
    ]
    activation = (
        getattr(nn, activation) if activation is not None else nn.Identity
    )
    activations = [activation() for _ in hidden_shape] + [nn.Identity()]

    return torch.nn.Sequential(*list(chain(*zip(layers, activations))))


class TransformModule(nn.Module):
    def __init__(
        self,
        n_params: int,
        net_hidden_shape: Optional[list[int]] = None,
        net_activation: Optional[str] = None,
    ):
        super().__init__()
        if net_hidden_shape is None:
            self.register_parameter(
                "_params",
                nn.Parameter(torch.rand(1, n_params)),
            )
            # NOTE: don't use lambda as it can't be pickled
            self._get_params = self._get_params_unconditional
        else:
            self.register_module(
                "_net",
                make_fnn(
                    size_out=n_params,
                    hidden_shape=net_hidden_shape,
                    activation=net_activation,
                ),
            )
            self._get_params = self._get_params_conditional

    def _get_params_unconditional(self, k: Tensor) -> Tensor:
        return self._params

    def _get_params_conditional(self, k: Tensor) -> Tensor:
        return self._net(k)

    def params(self, k: Tensor | None) -> Tensor:
        return self._get_params(k)

    def forward(
        self, k: Tensor | None = None
    ) -> Callable[Tensor, tuple[Tensor, Tensor]]:
        ...
