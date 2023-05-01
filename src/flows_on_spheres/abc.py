from abc import ABC, abstractmethod
from typing import TypeAlias

import torch

Tensor: TypeAlias = torch.Tensor
Module: TypeAlias = torch.nn.Module


class Density(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @abstractmethod
    def density(self, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def log_density(self, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def grad_density(self, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def grad_log_density(self, x: Tensor) -> Tensor:
        ...


class Flow(Module, ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @abstractmethod
    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        ...


class Hamiltonian(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @abstractmethod
    def hamiltonian(self, x: Tensor, p: Tensor) -> Tensor:
        ...

    @abstractmethod
    def grad_wrt_p(self, p: Tensor) -> Tensor:
        ...

    @abstractmethod
    def grad_wrt_x(self, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def sample_momentum(self, x: Tensor) -> Tensor:
        ...


class Transformer(ABC):
    @property
    @abstractmethod
    def identity_params(self) -> Tensor:
        ...

    @property
    @abstractmethod
    def n_params(self) -> int:
        ...

    @abstractmethod
    def forward(self, inputs: Tensor, params: Tensor) -> tuple[Tensor, Tensor]:
        ...

    def __call__(
        self, inputs: Tensor, params: Tensor
    ) -> tuple[Tensor, Tensor]:
        return self.forward(inputs, params)
