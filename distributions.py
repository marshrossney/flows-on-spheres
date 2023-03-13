import math

import torch

Tensor = torch.Tensor

PI = math.pi


class SphericalUniformPrior3D(torch.utils.data.IterableDataset):
    def __init__(self, sample_shape: int | list[int]):
        super().__init__()
        if not hasattr(sample_shape, "__iter__"):
            sample_shape = [sample_shape]
        if len(sample_shape) < 2:
            sample_shape += [1]
        self.shape = sample_shape
        self.log_surface_area = math.log(4 * PI)

    def __iter__(self):
        return self

    def __next__(self):
        theta = torch.acos(1 - 2 * torch.rand(self.shape))
        phi = torch.rand(self.shape) * 2 * PI

        x = theta.sin() * phi.cos()
        y = theta.sin() * phi.sin()
        z = theta.cos()

        outputs = torch.stack([x, y, z], dim=-1)
        logq = (
            torch.full(
                self.shape, fill_value=-self.log_surface_area, device=outputs.device
            )
            .flatten(start_dim=1)
            .sum(dim=1)
        )

        return outputs, logq


def marsaglia(N: int, size: torch.Size = torch.Size([1])) -> Tensor:
    u = torch.empty(*size, N, dtype=torch.float64).normal_()
    x = u.div(torch.linalg.vector_norm(u, dim=-1, keepdim=True))

    # Drop nans and infs
    isfinite = x.isfinite().flatten(start_dim=1).all(dim=1)
    x = x[isfinite]

    return x.float()


class SphericalUniformPrior(torch.utils.data.IterableDataset):
    def __init__(self, N: int, sample_shape: int | list[int]):
        super().__init__()
        if not hasattr(sample_shape, "__iter__"):
            sample_shape = [sample_shape]
        if len(sample_shape) < 2:
            sample_shape += [1]
        self.shape = sample_shape
        self.N = N
        self.log_surface_area = (
            math.log(2) + (N / 2) * math.log(PI) - math.lgamma(N / 2)
        )

    def __iter__(self):
        return self

    def __next__(self):
        x = marsaglia(self.N, self.shape)
        logq = (
            torch.full(self.shape, -self.log_surface_area, device=x.device)
            .flatten(start_dim=1)
            .sum(dim=1)
        )
        return x, logq


class vonMisesFisherPrior3D(torch.utils.data.IterableDataset):
    def __init__(self, batch_size: int, kappa: float):
        super().__init__()
        self.batch_size = batch_size
        self.kappa = kappa

    def __iter__(self):
        return self

    def __next__(self):
        xi = torch.empty(N).uniform_(0, 1)
        phi = torch.empty(N).uniform_(0, 2 * PI)

        z = (
            torch.log(math.exp(-self.kappa) + 2 * xi * math.sinh(self.kappa))
            / self.kappa
        )

        rho = (1 - z**2).clamp(min=1e-8).sqrt()
        x = rho * phi.cos()
        y = rho * phi.sin()

        logq = z - 1 + math.log(self.kappa / (2 * PI * (1 - math.exp(-2 * self.kappa))))

        return torch.stack([x, y, z], dim=1), logq
