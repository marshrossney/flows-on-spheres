from math import pi as π

from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import torch

from flows_on_spheres.distributions import marsaglia
from flows_on_spheres.transforms import ExponentialMapTransform
from flows_on_spheres.geometry import sphere_vectors_to_angles

def scatter(coords, data, name):

    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    cmap = ScalarMappable(cmap="viridis")
    c = cmap.to_rgba(data)

    lat, lon = sphere_vectors_to_angles(coords).split(1, dim=-1)
    lon = lon.add(π).remainder(2 * π).sub(π)

    ax.scatter(lon, lat, s=1, c=c)
    fig.colorbar(cmap, ax=ax)
    fig.tight_layout()

    fig.savefig(f"{name}.png", dpi=200, bbox_inches="tight")

D = 2
N = 100000
n = 2
layers = 1

f = ExponentialMapTransform(dim=D, n_mixture=n)

x = marsaglia(D, N)
mu = marsaglia(D, n).view(1, n, D + 1)

mu = torch.stack(
        [
            torch.tensor([0, 1, 0]),
            torch.tensor([0, -1, 0]),
            #torch.tensor([0, 0, 1]),
            #torch.tensor([0, 0, -1]),
        ],
        dim=0
).unsqueeze(0)

#kappa = torch.empty((1, n, 1)).uniform_()
kappa = torch.full((1, n, 1), 8)
rho = torch.ones((1, n, 1))

params = torch.cat([rho, kappa, mu], dim=-1).expand(N, -1, -1)

y = x
ldj = 0
for layer in range(layers):
    outputs = f(y, params)
    y = outputs["y"]
    ldj += outputs["dlV"]

scatter(outputs["x"], outputs["norm_v"], "x_vs_norm_v")

