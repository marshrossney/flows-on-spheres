from abc import ABCMeta, abstractmethod
from itertools import chain
from typing import Callable, Optional, TypeAlias
import warnings

import torch
import torch.nn as nn

Tensor: TypeAlias = torch.Tensor

warnings.filterwarnings("ignore", message="Lazy modules are a new feature")

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


class _TransformModule(nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        n_params: int,
        net_hidden_shape: Optional[list[int]] = None,
        net_activation: Optional[str] = None,
    ):
        super().__init__()
        assert isinstance(n_params, int)
        assert n_params > 0
        if net_hidden_shape is None:
            self.register_parameter(
                "_params",
                nn.Parameter(torch.rand(n_params)),
            )
            self._forward = self._forward_unconditional

        else:
            self.register_module(
                "_net",
                make_fnn(
                    size_out=n_params,
                    hidden_shape=net_hidden_shape,
                    activation=net_activation,
                ),
            )
            self._forward = self._forward_conditional

    def _forward_conditional(self, k: Tensor):
        assert isinstance(k, Tensor)
        params = self._net(k)
        transform = self.transform(params)
        return transform

    def _forward_unconditional(self, k: None):
        assert k is None
        params = self._params
        transform = self.transform(params)
        return transform

    def forward(
        self, k: Tensor | None
    ) -> Callable[Tensor, tuple[Tensor, Tensor]]:
        return self._forward(k)

    @abstractmethod
    def transform(
        self, params: Tensor
    ) -> Callable[Tensor, tuple[Tensor, Tensor]]:
        ...


class TransformModule(_TransformModule):
    pass


class CircularTransformModule(_TransformModule):
    pass
