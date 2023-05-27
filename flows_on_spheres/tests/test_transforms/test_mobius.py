from math import pi as π

import pytest
import torch
from torch import allclose

from flows_on_spheres.transforms.mobius import (
    _MobiusTransform,
    _MobiusMixtureTransform,
    MobiusModule,
)
from flows_on_spheres.linalg import norm

SAMPLE_SIZE = 100


@pytest.fixture
def inputs():
    θ = torch.empty(SAMPLE_SIZE, 1).uniform_(0, 2 * π)
    x = torch.cat([θ.cos(), θ.sin()], dim=1)
    return x


def test_identity(inputs):
    x = inputs
    f = _MobiusTransform(torch.zeros_like(x))
    y, ldj = f(x)
    assert allclose(x, y)
    assert allclose(ldj, torch.zeros(1), atol=1e-5)


@pytest.mark.xfail(reason="fails with large samples. Unsure exactly why")
def test_identity_mixture(inputs):
    x = inputs
    omega = torch.zeros(SAMPLE_SIZE, 3, 2)
    f = _MobiusMixtureTransform(omega, None)
    y, ldj = f(x)
    assert allclose(x, y)
    assert allclose(ldj, torch.zeros(1), atol=1e5)


# Testing MobiusModule


def test_single(inputs):
    x = inputs
    transform = MobiusModule(
        n_mixture=1,
        weighted=False,
        net_hidden_shape=None,
        net_activation=None,
    )
    y, ldj = transform(None)(x)
    assert allclose(norm(y), torch.ones(1))


def test_roundtrip(inputs):
    x = inputs
    transform = MobiusModule(
        n_mixture=1,
        weighted=False,
        net_hidden_shape=None,
        net_activation=None,
    )
    f = transform(None)
    y, ldj = f(x)
    z, ldj_inv = f.inverse(y)
    assert allclose(x, z, atol=1e-5)
    assert allclose(ldj + ldj_inv, torch.zeros(1), atol=1e-5)


def test_mixture_2(inputs):
    x = inputs
    transform = MobiusModule(
        n_mixture=2,
        weighted=False,
        net_hidden_shape=None,
        net_activation=None,
    )
    y, ldj = transform(None)(x)
    assert allclose(norm(y), torch.ones(1))


def test_mixture_3(inputs):
    x = inputs
    transform = MobiusModule(
        n_mixture=3,
        weighted=False,
        net_hidden_shape=None,
        net_activation=None,
    )
    y, ldj = transform(None)(x)
    assert allclose(norm(y), torch.ones(1))


def test_weighted_mixture_2(inputs):
    x = inputs
    transform = MobiusModule(
        n_mixture=2,
        weighted=True,
        net_hidden_shape=None,
        net_activation=None,
    )
    y, ldj = transform(None)(x)
    assert allclose(norm(y), torch.ones(1))


def test_weighted_mixture_3(inputs):
    x = inputs
    transform = MobiusModule(
        n_mixture=3,
        weighted=True,
        net_hidden_shape=None,
        net_activation=None,
    )
    y, ldj = transform(None)(x)
    assert allclose(norm(y), torch.ones(1))


def test_conditional(inputs):
    x = inputs
    transform = MobiusModule(
        n_mixture=1,
        weighted=False,
        net_hidden_shape=[],
        net_activation=None,
    )
    k = torch.rand_like(x)
    y, ldj = transform(k)(x)


def test_conditional_weighted_mixture(inputs):
    x = inputs
    transform = MobiusModule(
        n_mixture=3,
        weighted=True,
        net_hidden_shape=[],
        net_activation=None,
    )
    k = torch.rand_like(x)
    y, ldj = transform(k)(x)
