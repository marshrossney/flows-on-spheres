from math import pi as π

import pytest
import torch
from torch import allclose

from flows_on_spheres.transforms.sigmoid import (
    _BumpTransform,
    BumpModule,
)
from flows_on_spheres.linalg import norm

SAMPLE_SIZE = 100


@pytest.fixture
def interval_inputs():
    return torch.empty(SAMPLE_SIZE, 1).uniform_(-1, 1)


@pytest.fixture
def circular_inputs():
    θ = torch.empty(SAMPLE_SIZE, 1).uniform_(0, 2 * π)
    x = torch.cat([θ.cos(), θ.sin()], dim=1)
    return x


def test_identity(interval_inputs):
    x = interval_inputs
    f = _BumpTransform(
        torch.rand_like(x),
        torch.rand_like(x),
        torch.zeros_like(x),
        torch.rand_like(x),
    )
    y, ldj = f(x)
    assert allclose(x, y, atol=1e-5)
    assert allclose(ldj, torch.zeros(1), atol=1e-5)


def test_single(interval_inputs):
    x = interval_inputs
    transform = BumpModule(
        n_mixture=1,
        weighted=False,
        net_hidden_shape=None,
        net_activation=None,
    )
    y, ldj = transform(None)(x)
    assert y.isfinite().all()
    assert ldj.isfinite().all()
