from math import pi as π

import torch
import pytest

from flows_on_spheres.transforms import MobiusModule
from flows_on_spheres.flows import CircularFlow

SAMPLE_SIZE = 100


@pytest.fixture
def inputs():
    θ = torch.empty(SAMPLE_SIZE, 1).uniform_(0, 2 * π)
    x = torch.cat([θ.cos(), θ.sin()], dim=1)
    return x


def test_mobius(inputs):
    x = inputs

    transform = MobiusModule(
        n_mixture=2,
        weighted=True,
        net_hidden_shape=None,
        net_activation=None,
    )

    flow = CircularFlow([transform])

    y, ldj = flow(x)
