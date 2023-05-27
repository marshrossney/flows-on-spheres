import pytest
import torch

from flows_on_spheres.flows import RecursiveFlowS2
from flows_on_spheres.transforms import MobiusModule, RQSplineModule

SAMPLE_SIZE = 100


@pytest.fixture
def inputs():
    x = torch.randn(SAMPLE_SIZE, 3)
    x = x / x.pow(2).sum(1, keepdim=True).sqrt()
    return x


def test_runs(inputs):
    x = inputs

    z_transform = RQSplineModule(
        n_segments=4,
        circular=False,
        net_hidden_shape=None,
        net_activation=None,
    )
    xy_transform = MobiusModule(
        n_mixture=1,
        weighted=False,
        net_hidden_shape=[],
        net_activation=None,
    )

    flow = RecursiveFlowS2([z_transform], [xy_transform])

    y, ldj = flow(x)
