from math import pi as π
from typing import TypeAlias, Optional

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.linalg as LA

from vonmises.utils import circle_vectors_to_angles

#from utils import spherical_mesh

Tensor: TypeAlias = torch.Tensor
Figure: TypeAlias = plt.Figure

sns.set_theme()



def scatter2d(xy: Tensor, polar: bool = False, **scatter_kwargs) -> Figure:
    if polar:
        r = LA.vector_norm(xy, dim=-1, keepdim=True)
        ϕ = circle_vectors_to_angles(xy)
        fig, ax = plt.subplots(subplot_kw={"polar": True})
        ax.set_ylim([0, 1.1 * r.max()])
        ax.set_xticklabels([])
        ax.set_aspect("equal")
        ax.scatter(ϕ, r, **scatter_kwargs)
        return fig
    else:
        x, y = xy.split(1, dim=-1)
        fig, ax = plt.subplots()
        ax.scatter(x, y, **scatter_kwargs)
        return fig


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


def line(xyz: Tensor, projection: str = "aitoff", **plot_kwargs) -> plt.Figure:
    x, y, z = xyz.split(1, dim=-1)
    θ = torch.asin(z)
    ϕ = torch.atan2(y, x)

    fig, ax = plt.subplots(subplot_kw=dict(projection=projection))
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    ax.plot(ϕ, θ, **plot_kwargs)

    fig.tight_layout()

    return fig



def scatter(
    xyz: Tensor,
    colours: Optional[Tensor] = None,
    projection: str = "aitoff",
    **scatter_kwargs
) -> plt.Figure:
    x, y, z = xyz.split(1, dim=-1)
    θ = torch.asin(z)
    ϕ = torch.atan2(y, x)

    if colours is not None:
        cmap = ScalarMappable(
            norm=Normalize(vmin=colours.min(), vmax=colours.max()), cmap="viridis"
        )
        colours = cmap.to_rgba(colours)

    fig, ax = plt.subplots(subplot_kw=dict(projection=projection))
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    sc = ax.scatter(ϕ, θ, c=colours, **scatter_kwargs)

    fig.colorbar(cmap, ax=ax, shrink=1)

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
