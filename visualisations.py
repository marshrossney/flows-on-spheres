from math import pi as π
from typing import TypeAlias

import numpy as np
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
    [(ax.set_xlim(-1.1, 1.1), ax.set_ylim(-1.1, 1.1)) for ax in grid.axes.flatten() if ax is not None]
    
    return grid

def heatmap(xyz: Tensor, bins: int = 50, **kwargs) -> plt.Figure:
    
    x, y, z = xyz.flatten(end_dim=-2).split(1, dim=-1) # don't care about batch_size
    θ = torch.asin(z).squeeze().tolist()
    ϕ = torch.atan2(y, x).squeeze().tolist()
    
    lon = np.linspace(-π, π, bins + 1)
    lat = np.linspace(-π/2, π/2, bins + 1)
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="mollweide")
    
    hist, lon, lat = np.histogram2d(ϕ, θ, bins=[lon, lat], density=True)
        
    ax.pcolormesh(
        lon[:-1], lat[:-1],
        hist.T,
        cmap="viridis",
        shading='auto',
        vmin=0,
        vmax=1
    )
        
    fig.tight_layout()
    
    return fig

def scatter(xyz: Tensor, axis: bool = True, **kwargs) -> plt.Figure:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(*spherical_mesh(25), rstride=1, cstride=1, alpha=0.2, lw=0.1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    if not axis:
        ax.set_axis_off()
    
    xyz = xyz.unsqueeze(dim=0).flatten(start_dim=0, end_dim=-3)

    for batch in xyz.split(1, dim=0):
        x, y, z = batch.split(1, dim=-1)
        ax.scatter3D(x.squeeze(), y.squeeze(), z.squeeze(), **kwargs)
        
    fig.tight_layout()
    
    return fig