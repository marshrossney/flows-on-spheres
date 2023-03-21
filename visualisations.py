from math import pi as π
from typing import TypeAlias

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from utils import spherical_mesh

Tensor: TypeAlias = torch.Tensor

sns.set_theme()


def pairplot(xyz: Tensor) -> sns.PairGrid:
    df = pd.DataFrame(xyz, columns=["x", "y", "z"])

    grid = sns.pairplot(df, kind="hist", corner=True)
    [
        (ax.set_xlim(-1.1, 1.1), ax.set_ylim(-1.1, 1.1))
        for ax in grid.axes.flatten()
        if ax is not None
    ]

    return grid


def heatmap(xyz: Tensor, bins: int = 50, **pcolormesh_kwargs) -> plt.Figure:
    x, y, z = xyz.flatten(end_dim=-2).split(1, dim=-1)  # merges all batches
    θϕ = torch.cat([torch.asin(z), torch.atan2(y, x)], dim=-1)
    θ_grid = torch.linspace(-π / 2, π / 2, bins + 1)
    ϕ_grid = torch.linspace(-π, π, bins + 1)

    hist, _ = torch.histogramdd(θϕ, bins=[θ_grid, ϕ_grid], density=True)

    fig, ax = plt.subplots(subplot_kw=dict(projection="mollweide"))
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    pcolormesh_kwargs = (
        dict(cmap="viridis", shading="auto", vmin=0, vmax=1) | pcolormesh_kwargs
    )

    ax.pcolormesh(
        ϕ_grid,
        θ_grid,
        hist,
        **pcolormesh_kwargs,
    )

    fig.tight_layout()

    return fig


def line(xyz: Tensor, figsize: tuple = (6, 6), **plot_kwargs) -> plt.Figure:
    x, y, z = xyz.split(1, dim=-1)
    θ = torch.asin(z)
    ϕ = torch.atan2(y, x)
    θϕ = torch.cat([θ, ϕ], dim=-1)

    fig, ax = plt.subplots(subplot_kw=dict(projection="mollweide"))
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    ax.plot(ϕ, θ, **plot_kwargs)

    fig.tight_layout()

    return fig


def scatter(xyz: Tensor, figsize: tuple = (6, 6), **scatter_kwargs) -> plt.Figure:
    x, y, z = xyz.split(1, dim=-1)
    θ = torch.asin(z)
    ϕ = torch.atan2(y, x)
    θϕ = torch.cat([θ, ϕ], dim=-1)

    fig, ax = plt.subplots(subplot_kw=dict(projection="mollweide"))
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    ax.scatter(ϕ, θ, **scatter_kwargs)

    fig.tight_layout()

    return fig


def scatter3d(xyz: Tensor, **scatter3D_kwargs) -> plt.Figure:
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    ax.plot_surface(*spherical_mesh(25), rstride=1, cstride=1, alpha=0.2, lw=0.1)

    xyz = xyz.unsqueeze(dim=0).flatten(start_dim=0, end_dim=-3)
    for batch in xyz.split(1, dim=0):
        x, y, z = batch.split(1, dim=-1)
        ax.scatter3D(x.squeeze(), y.squeeze(), z.squeeze(), **scatter3D_kwargs)

    ax.set_axis_off()
    fig.tight_layout()

    return fig


def line3d(xyz: Tensor, hide_axis: bool = True, **plot3D_kwargs) -> plt.Figure:
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

    ax.plot_surface(*spherical_mesh(25), rstride=1, cstride=1, alpha=0.2, lw=0.1)

    xyz = xyz.unsqueeze(dim=0).flatten(start_dim=0, end_dim=-3)
    for batch in xyz.split(1, dim=0):
        x, y, z = batch.split(1, dim=-1)
        ax.plot3D(
            x.squeeze().tolist(),
            y.squeeze().tolist(),
            z.squeeze().tolist(),
            **plot3D_kwargs,
        )

    ax.set_axis_off()
    fig.tight_layout()

    return fig
