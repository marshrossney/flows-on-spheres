from abc import ABC, abstractmethod
from math import pi as π
from typing import TypeAlias, Optional, Generator

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.linalg as LA

from flows_on_spheres.model import FlowBasedModel
from flows_on_spheres.geometry import (
    spherical_mesh,
    circle_vectors_to_angles,
    circle_angles_to_vectors,
    sphere_vectors_to_angles,
)
from flows_on_spheres.hmc import add_fhmc_hooks

Tensor: TypeAlias = torch.Tensor
Figure: TypeAlias = plt.Figure
Axes: TypeAlias = plt.Axes

sns.set_theme()


class Visualiser(ABC):
    @abstractmethod
    def __init__(self, model: FlowBasedModel, *args, **kwargs):
        ...

    @abstractmethod
    def figures(self) -> Generator[tuple[str, Figure], None, None]:
        ...


class CircularFlowVisualiser(Visualiser):
    def __init__(
        self,
        model: FlowBasedModel,
        sample_size: int = int(1e6),
    ):
        self.model = model.to(device="cpu")

        hooks = add_fhmc_hooks(self.model.flow, self.model.target)
        sample = self.model.sample(sample_size)
        [hook.remove() for hook in hooks]

        self._sample = {key: tensor.detach() for key, tensor in sample.items()}
        self._sample["forces"] = sample["inputs"].grad

        self._sample["input_angles"] = circle_vectors_to_angles(
            self._sample["inputs"]
        )
        self._sample["output_angles"] = circle_vectors_to_angles(
            self._sample["outputs"]
        )

        self._linspace_angles = torch.linspace(0, 2 * π, 1000).unsqueeze(-1)
        self._linspace_target_density = self.model.target.density(
            circle_angles_to_vectors(self._linspace_angles)
        )

    def _get_figure(self) -> tuple[Figure, Axes, Axes]:
        fig = plt.figure(figsize=(7, 4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, projection="polar")
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        return fig, ax1, ax2

    def transform(self) -> Figure:
        fig, ax1, ax2 = self._get_figure()
        ax1.scatter(
            self._sample["input_angles"], self._sample["output_angles"], s=0.3
        )
        ax2.scatter(
            self._sample["input_angles"], self._sample["output_angles"], s=0.3
        )
        ax1.set_xlabel(r"$\phi_{in}$")
        ax1.set_ylabel(r"$\phi_{out}$")
        fig.suptitle("Transformation")
        fig.tight_layout()
        return fig

    def histogram(self, bins: int = 36) -> Figure:
        hist, bin_edges = torch.histogram(
            self._sample["output_angles"],
            bins=bins,
            range=(0, 2 * π),
            density=True,
        )
        positions = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        fig, ax1, ax2 = self._get_figure()
        ax1.bar(
            positions,
            hist,
            width=(2 * π) / bins,
            color="tab:blue",
            zorder=1,
            label="data",
        )
        ax1.plot(
            self._linspace_angles,
            self._linspace_target_density,
            color="tab:orange",
            zorder=2,
            label="target",
        )
        ax2.bar(
            positions,
            hist,
            width=(2 * π) / bins,
            bottom=0.01,
            color="tab:blue",
            zorder=1,
        )
        ax2.plot(
            self._linspace_angles,
            self._linspace_target_density,
            color="tab:orange",
            zorder=2,
        )
        ax1.set_xlabel(r"$\phi_{out}$")
        ax1.set_ylabel(r"density")
        fig.suptitle("Histogram of generated data")
        fig.legend()
        fig.tight_layout()
        return fig

    def log_density(self) -> Figure:
        fig, ax1, ax2 = self._get_figure()
        ax1.plot(
            self._linspace_angles,
            self._linspace_target_density.log(),
            color="tab:orange",
            zorder=1,
            label="target",
        )
        ax1.scatter(
            self._sample["output_angles"],
            self._sample["log_density"],
            s=0.3,
            c="tab:blue",
            zorder=2,
            label="model",
        )
        ax2.plot(
            self._linspace_angles,
            self._linspace_target_density.log(),
            color="tab:orange",
            zorder=1,
        )
        ax2.scatter(
            self._sample["output_angles"],
            self._sample["log_density"],
            s=0.3,
            c="tab:blue",
            zorder=2,
        )
        ax1.set_xlabel(r"$\phi_{out}$")
        ax1.set_ylabel(r"log density")
        fig.suptitle("Model and target densities")
        fig.legend()
        fig.tight_layout()
        return fig

    def log_weights(self) -> Figure:
        fig, ax1, ax2 = self._get_figure()
        ax1.scatter(
            self._sample["output_angles"],
            self._sample["log_weights"],
            s=0.3,
            zorder=2,
        )
        ax1.axhline(y=0, color="tab:orange", zorder=1)
        ax2.scatter(
            self._sample["output_angles"],
            self._sample["log_weights"],
            s=0.3,
            zorder=2,
        )
        ax2.plot(
            self._linspace_angles,
            torch.zeros_like(self._linspace_angles),
            color="tab:orange",
            zorder=1,
        )
        ax1.set_xlabel(r"$\phi_{out}$")
        ax1.set_ylabel(r"log weights")
        fig.suptitle("Statistical weights for model outputs")
        fig.tight_layout()
        return fig

    def forces(self) -> Figure:
        fig, ax1, ax2 = self._get_figure()
        f1, f2 = self._sample["forces"].split(1, -1)
        ax1.scatter(
            self._sample["output_angles"],
            f1,
            s=0.3,
            c="tab:blue",
            zorder=2,
            label="$f_1$",
        )
        ax1.scatter(
            self._sample["output_angles"],
            f2,
            s=0.3,
            c="tab:orange",
            zorder=3,
            label="$f_2$",
        )
        ax1.axhline(y=0, color="black", zorder=1)
        ax2.scatter(
            self._sample["output_angles"], f1, s=0.3, zorder=2, c="tab:blue"
        )
        ax2.scatter(
            self._sample["output_angles"], f2, s=0.3, zorder=3, c="tab:orange"
        )
        ax2.plot(
            self._linspace_angles,
            torch.zeros_like(self._linspace_angles),
            color="black",
            zorder=1,
        )
        ax1.set_xlabel(r"$\phi$ (outputs)")
        ax1.set_ylabel(r"$-\nabla S_{eff}$")
        fig.suptitle("HMC Forces")
        fig.legend()
        fig.tight_layout()
        return fig

    def figures(self) -> Generator[tuple[str, Figure], None, None]:
        yield "histogram", self.histogram()
        yield "transform", self.transform()
        yield "log_density", self.log_density()
        yield "log_weights", self.log_weights()
        yield "forces", self.forces()


class SphericalFlowVisualiser(Visualiser):
    def __init__(
        self,
        model: FlowBasedModel,
        sample_size: int = int(1e5),
        projection: str = "aitoff",
    ):
        assert model.target.dim == 2

        self.model = model.to(device="cpu")
        self.projection = projection

        hooks = add_fhmc_hooks(self.model.flow, self.model.target)
        sample = self.model.sample(sample_size)
        [hook.remove() for hook in hooks]

        self._sample = {key: tensor.detach() for key, tensor in sample.items()}
        self._sample["forces"] = sample["inputs"].grad
        self._sample["input_angles"] = sphere_vectors_to_angles(
            self._sample["inputs"]
        )
        self._sample["output_angles"] = sphere_vectors_to_angles(
            self._sample["outputs"]
        )

    def _make_scatter(
        self,
        data: Tensor,
        title: str,
        coords: str = "outputs",
        topk: Optional[int] = None,
    ) -> Figure:
        assert coords in ("inputs", "outputs")
        coords = self._sample[
            "input_angles" if coords == "inputs" else "output_angles"
        ]
        lat, lon = coords.split(1, dim=-1)
        lon = lon.add(π).remainder(2 * π).sub(π)
        # lat = -(lat - π / 2)

        if topk is not None:
            data_topk, indices_topk = torch.topk(
                data, k=topk, largest=True, sorted=True
            )

            lat_topk = lat[indices_topk]
            lon_topk = lon[indices_topk]

            topk_mask = torch.ones_like(data).bool()
            topk_mask[indices_topk] = False

            data = data[topk_mask]
            lat = lat[topk_mask]
            lon = lon[topk_mask]

        cmap = ScalarMappable(
            norm=Normalize(vmin=data.min(), vmax=data.max()),
            cmap="viridis",
        )

        fig, ax = plt.subplots(subplot_kw=dict(projection=self.projection))
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        ax.scatter(lon, lat, s=3, c=cmap.to_rgba(data))

        if topk is not None:
            ax.scatter(
                lon_topk, lat_topk, s=3, c="red", label=f"Largest {topk}"
            )
            ax.legend()

        ax.set_title(title)
        fig.colorbar(cmap, ax=ax, shrink=0.55)
        fig.tight_layout()

        return fig

    def density(
        self,
        coords: str = "outputs",
        negative: bool = False,
        topk: Optional[int] = None,
    ) -> Figure:
        data = self._sample["log_density"].exp()
        if negative:
            data = -data
            title = "Negative model density: " + r"$-\tilde{p}_f$"
        else:
            title = "Model density: " + r"$\tilde{p}_f$"
        return self._make_scatter(data, title, coords, topk=topk)

    def log_density(
        self,
        coords: str = "outputs",
        negative: bool = False,
        topk: Optional[int] = None,
    ) -> Figure:
        data = self._sample["log_density"]
        if negative:
            data = -data
            title = "Negative log model density: " + r"$-\log \tilde{p}_f$"
        else:
            title = "Log model density: " + r"$\log \tilde{p}_f$"
        return self._make_scatter(data, title, coords, topk=topk)

    def log_weights(
        self,
        coords: str = "outputs",
        negative: bool = False,
        topk: Optional[int] = None,
    ) -> Figure:
        data = self._sample["log_weights"]
        if negative:
            data = -data
            title = "Negative log weights: " + r"$\log \tilde{p}_f - \log p$"
        else:
            title = "Log weights: " + r"$\log p - \log \tilde{p}_f$"
        return self._make_scatter(data, title, coords, topk=topk)

    def force(
        self,
        component: Optional[str] = None,
        coords: str = "outputs",
        topk: Optional[int] = None,
    ) -> Figure:
        if component is not None:
            i = dict(x=0, y=1, z=2)[component]
            data = self._sample["forces"][:, i].abs()
            title = (
                f"Force ({component} component): "
                + r"$\vert$"
                + f"$F_{component}$"
                + r"$\vert$"
            )
        else:
            data = LA.vector_norm(self._sample["forces"], dim=-1)
            title = "Force: " + r"$\vert \mathbf{F} \vert$"
        return self._make_scatter(data, title, coords, topk=topk)

    def forces_quiver(self, coords: str = "outputs", n: int = 1000) -> Figure:
        coords = self._sample[
            "input_angles" if coords == "inputs" else "output_angles"
        ]
        lat, lon = coords[:n].split(1, dim=-1)
        lon = lon.add(π).remainder(2 * π).sub(π)

        data = sphere_vectors_to_angles(self._sample["forces"])
        dlat, dlon = data[:n].split(1, dim=-1)
        dlon = dlon.add(π).remainder(2 * π).sub(π)

        colors = LA.vector_norm(self._sample["forces"], dim=-1)[:n]
        cmap = ScalarMappable(
            norm=Normalize(vmin=colors.min(), vmax=colors.max()),
            cmap="viridis",
        )
        colors = cmap.to_rgba(colors)

        fig, ax = plt.subplots(subplot_kw=dict(projection=self.projection))
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        ax.quiver(dlon, dlat, lon, lat, angles="xy", scale=100)

        ax.set_title("Forces")
        # fig.colorbar(cmap, ax=ax, shrink=0.55)
        fig.tight_layout()

        return fig

    def figures(self) -> Generator[tuple[str, Figure], None, None]:
        # TODO: not sure whether we care enough about this to make it
        # configurable via the cli
        topk = len(self._sample["inputs"]) // 100  # 1% of inputs

        for coords in ("inputs", "outputs"):
            yield f"density_{coords}", self.density(coords)
            yield f"log_density_{coords}", self.log_density(coords)
            yield f"log_weights_{coords}", self.log_weights(coords)
            yield f"negative_log_weights_{coords}", self.log_weights(
                coords, negative=True, topk=topk
            )
            yield f"total_force_{coords}", self.force(None, coords, topk=topk)
            yield f"x_force_{coords}", self.force("x", coords, topk=topk)
            yield f"y_force_{coords}", self.force("y", coords, topk=topk)
            yield f"z_force_{coords}", self.force("z", coords, topk=topk)
            # yield "forces_quiver", self.forces_quiver(coords)


def scatter3d(xyz: Tensor, **scatter3D_kwargs) -> plt.Figure:
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    ax.plot_surface(
        *spherical_mesh(25), rstride=1, cstride=1, alpha=0.2, lw=0.1
    )

    xyz = xyz.unsqueeze(dim=0).flatten(start_dim=0, end_dim=-3)
    for batch in xyz.split(1, dim=0):
        x, y, z = batch.split(1, dim=-1)
        ax.scatter3D(x.squeeze(), y.squeeze(), z.squeeze(), **scatter3D_kwargs)

    ax.set_axis_off()
    fig.tight_layout()

    return fig


def line3d(xyz: Tensor, hide_axis: bool = True, **plot3D_kwargs) -> plt.Figure:
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

    ax.plot_surface(
        *spherical_mesh(25), rstride=1, cstride=1, alpha=0.2, lw=0.1
    )

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
