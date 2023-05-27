import pytest
import torch

from flows_on_spheres.transforms.bspline import BSplineModule

SAMPLE_SIZE = 100


@pytest.fixture
def inputs():
    return torch.empty(SAMPLE_SIZE, 1).uniform_(-1, +1)


def test_forward(inputs):
    x = inputs
    transform = BSplineModule(
        n_intervals=4,
        net_hidden_shape=None,
        net_activation=None,
    )
    y, ldj = transform(None)(x)


def test_conditional(inputs):
    x = inputs
    transform = BSplineModule(
        n_intervals=4,
        net_hidden_shape=[],
        net_activation=None,
    )
    y, ldj = transform(torch.rand_like(x))(x)
