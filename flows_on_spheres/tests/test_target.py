from math import pi as π

import pytest
import torch

from flows_on_spheres.target import (
    VonMisesFisherDensity,
    VonMisesFisherMixtureDensity,
)
from flows_on_spheres.linalg import dot

N = 10


@pytest.fixture
def inputs_2d():
    θ = torch.empty(N, 1).uniform_(0, 2 * π)
    x = torch.cat([θ.cos(), θ.sin()], dim=1)
    return x


def test_unimodal_2d(inputs_2d):
    x = inputs_2d
    target = VonMisesFisherDensity(10, (0, 1))

    _ = target.density(x)
    _ = target.log_density(x)
    gd = target.grad_density(x)
    gld = target.grad_log_density(x)

    assert gd.shape == (N, 2)
    print(dot(gd, x))

    assert gld.shape == (N, 2)
    print(dot(gld, x))
