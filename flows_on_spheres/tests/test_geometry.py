from math import pi as π

import pytest
import torch

from flows_on_spheres.geometry import *

SAMPLE_SIZE = 100000


@pytest.fixture
def s1_angles():
    return torch.empty(SAMPLE_SIZE, 1).uniform_(0, 2 * π)


@pytest.fixture
def s1_vectors(s1_angles):
    return torch.cat([s1_angles.cos(), s1_angles.sin()], dim=1)


def test_angle_vector_conversion_s1(s1_angles, s1_vectors):
    # atol 1e-7 fails!
    assert torch.allclose(as_vector(as_angle(s1_vectors)), s1_vectors, atol=1e-6, rtol=0)
    assert torch.allclose(as_angle(as_vector(s1_angles)), s1_angles, atol=1e-6, rtol=0)
