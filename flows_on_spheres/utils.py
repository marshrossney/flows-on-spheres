from math import pi as π

import torch

Tensor = torch.Tensor


def mod_2pi(angles: Tensor) -> Tensor:
    return torch.remainder(angles, 2 * π)
