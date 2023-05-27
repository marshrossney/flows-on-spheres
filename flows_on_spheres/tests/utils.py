import torch
import torch.nn as nn

from flows_on_spheres.geometry import rotate_2d

Tensor = torch.Tensor


class DummyFlow(nn.Module):
    dim = None

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        return inputs, inputs.new_zeros(len(inputs))


class Rotation2D(nn.Module):
    dim = 1

    def __init__(self, angle: float):
        super().__init__()
        self.register_buffer("angle", torch.tensor(angle, dtype=torch.float))

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        outputs = rotate_2d(inputs, self.angle)
        return outputs, inputs.new_zeros(len(inputs))
