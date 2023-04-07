from math import log, pi as π

from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from flows_on_spheres.distributions import marsaglia
from flows_on_spheres.transforms import ExponentialMapTransform
from flows_on_spheres.geometry import sphere_vectors_to_angles

sns.set_theme()

def scatter(coords, data, xlabel, clabel, name):

    fig, ax = plt.subplots(subplot_kw={"projection": "aitoff"})
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    cmap = ScalarMappable(cmap="viridis")
    c = cmap.to_rgba(data)

    lat, lon = sphere_vectors_to_angles(coords).split(1, dim=-1)
    lon = lon.add(π).remainder(2 * π).sub(π)

    ax.scatter(lon, lat, s=1, c=c)
    ax.set_xlabel(xlabel)
    cbar = fig.colorbar(cmap, ax=ax, shrink=0.55)
    cbar.set_label(clabel)
    fig.tight_layout()

    ax.set_title(f"Example gradient map transform\n{xlabel} vs {clabel}")
    fig.savefig(f"{name}.png", dpi=200, bbox_inches="tight")

D = 2
N = 100000
n = 10
layers = 1

f = ExponentialMapTransform(dim=D, n_mixture=n)

x = marsaglia(D, N)

mu = torch.stack(
        [
            #torch.tensor([1, 0, 0])
            torch.tensor([0, 1, 0]),
            torch.tensor([0, -1, 0]),
            torch.tensor([0, 0, 1]),
            torch.tensor([0, 0, -1]),
        ],
        dim=0
).unsqueeze(0)
mu = marsaglia(D, n).view(1, n, D + 1)

kappa = torch.empty((1, n, 1)).uniform_(-10, 5)
#kappa = torch.full((1, n, 1), -5)
rho = torch.ones((1, n, 1))

params = torch.cat([rho, kappa, mu], dim=-1).expand(N, -1, -1)

y = x
ldj = 0
for layer in range(layers):
    outputs = f(y, params)
    y = outputs["y"]
    ldj += outputs["dlV"]

outputs["x"] = x
#outputs["log_density"] = log(4 * π) - outputs["dlV"]

scatter(outputs["x"], outputs["norm_v"], r"$\mathbf{x}$", r"$|\mathbf{v}|$", "inputs_vs_normv")
scatter(outputs["y"], outputs["norm_v"], r"$\mathbf{y}$", r"$|\mathbf{v}|$", "outputs_vs_normv")
scatter(outputs["x"], outputs["dlV"], r"$\mathbf{x}$", r"$\log \mathcal{V}$", "inputs_vs_logvol")
scatter(outputs["y"], outputs["dlV"], r"$\mathbf{y}$", r"$\log \mathcal{V}$", "outputs_vs_logvol")
scatter(outputs["x"], (outputs["x"] * outputs["y"]).sum(dim=-1), r"$\mathbf{x}$", r"$\mathbf{x} \cdot \mathbf{y}$", "x_vs_xdoty")

#scatter(outputs["x"], outputs["dlV"], "x_vs_log_vol")
#scatter(outputs["y"], outputs["dlV"], "y_vs_log_vol")

