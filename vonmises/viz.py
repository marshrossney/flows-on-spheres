from math import log, pi as π
from typing import TypeAlias, Optional

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.linalg as LA

from vonmises.distributions import uniform_prior
from vonmises.model import FlowBasedModel
from vonmises.geometry import (
    spherical_mesh,
    circle_vectors_to_angles,
    circle_angles_to_vectors,
    sphere_vectors_to_angles,
    sphere_angles_to_vectors,
)
from vonmises.hmc import add_fhmc_hooks

Tensor: TypeAlias = torch.Tensor
Figure: TypeAlias = plt.Figure
Axes: TypeAlias = plt.Axes

sns.set_theme()


class FlowVisualiserBase:
    _figures: list[str]

    def __init__(
        self,
        model: FlowBasedModel,
        sample_size: int = int(1e6),
    ):
        self.model = model.to(device="cpu")

        hooks = add_fhmc_hooks(self.model, self.model.target)
        sample = self.model.sample(sample_size)
        [hook.remove() for hook in hooks]

        self._inputs = sample["inputs"].detach()
        self._outputs = sample["outputs"].detach()
        self._log_prior_density = sample["log_prior_density"]
        self._log_model_density = sample["log_model_density"].detach()
        self._log_target_density = sample["log_target_density"].detach()
        self._inputs_grad = sample["inputs"].grad

    def figures(self) -> tuple[str, Figure]:
        for figure in self._figures:
            yield figure, getattr(self, figure)()


class CircularFlowVisualiser(FlowVisualiserBase):
    _figures: list[str] = [
        "transform",
        "histogram",
        "densities",
        "weights",
        "forces",
    ]

    def __init__(
        self,
        model: FlowBasedModel,
        sample_size: int = int(1e6),
        histogram_bins: int = 36,
    ):
        assert model.target.dim == 1
        super().__init__(model, sample_size)

        self._input_angles = circle_vectors_to_angles(self._inputs)
        self._output_angles = circle_vectors_to_angles(self._outputs)

        self._linspace_angles = torch.linspace(0, 2 * π, 1000).unsqueeze(-1)
        self._linspace_target_density = self.model.target.density(
            circle_angles_to_vectors(self._linspace_angles)
        )

        self._hist, bin_edges = torch.histogram(
            self._output_angles,
            bins=histogram_bins,
            range=(0, 2 * π),
            density=True,
        )
        self._positions = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def _get_figure(self) -> tuple[Figure, Axes, Axes]:
        fig = plt.figure(figsize=(7, 4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, projection="polar")
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        return fig, ax1, ax2

    def transform(self) -> Figure:
        fig, ax1, ax2 = self._get_figure()
        ax1.scatter(self._input_angles, self._output_angles, s=0.3)
        ax2.scatter(self._input_angles, self._output_angles, s=0.3)
        ax1.set_xlabel(r"$\phi_{in}$")
        ax1.set_ylabel(r"$\phi_{out}$")
        fig.suptitle("Transformation")
        fig.tight_layout()
        return fig

    def histogram(self) -> Figure:
        fig, ax1, ax2 = self._get_figure()
        ax1.bar(
            self._positions,
            self._hist,
            width=(2 * π) / len(self._hist),
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
            self._positions,
            self._hist,
            width=(2 * π) / len(self._hist),
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

    def densities(self) -> Figure:
        fig, ax1, ax2 = self._get_figure()
        ax1.plot(
            self._linspace_angles,
            self._linspace_target_density.log(),
            color="tab:orange",
            zorder=1,
            label="target",
        )
        ax1.scatter(
            self._output_angles,
            self._log_model_density,
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
            self._output_angles,
            self._log_model_density,
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

    def weights(self) -> Figure:
        fig, ax1, ax2 = self._get_figure()
        log_weights = self._log_target_density - self._log_model_density
        ax1.scatter(
            self._output_angles,
            log_weights,
            s=0.3,
            zorder=2,
        )
        ax1.axhline(y=0, color="tab:orange", zorder=1)
        ax2.scatter(
            self._output_angles,
            log_weights,
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
        f1, f2 = self._inputs_grad.split(1, -1)
        ax1.scatter(
            self._output_angles,
            f1,
            s=0.3,
            c="tab:blue",
            zorder=2,
            label="$f_1$",
        )
        ax1.scatter(
            self._output_angles,
            f2,
            s=0.3,
            c="tab:orange",
            zorder=3,
            label="$f_2$",
        )
        ax1.axhline(y=0, color="tab:green", zorder=1)
        ax2.scatter(self._output_angles, f1, s=0.3, zorder=2, c="tab:blue")
        ax2.scatter(self._output_angles, f2, s=0.3, zorder=3, c="tab:orange")
        ax2.plot(
            self._linspace_angles,
            torch.zeros_like(self._linspace_angles),
            color="tab:green",
            zorder=1,
        )
        ax1.set_xlabel(r"$\phi$ (outputs)")
        ax1.set_ylabel(r"$-\nabla S_{eff}$")
        fig.suptitle("HMC Forces")
        fig.legend()
        fig.tight_layout()
        return fig


class SphericalFlowVisualiser:
    def __init__(
        self,
        model: FlowBasedModel,
        sample_size: int = int(1e5),
        grid_size: int = 50,
        projection: str = "aitoff",
    ):
        assert model.target.dim == 2

        self.model = model.to(device="cpu")

        # Generate a uniform spherical mesh and pass through the flow
        theta_grid, phi_grid = torch.meshgrid(
            torch.linspace(0, π, grid_size), torch.linspace(-π, π, grid_size)
        )
        inputs = sphere_angles_to_vectors(
            torch.stack([theta_grid, phi_grid], dim=-1)
        ).view(grid_size**2, 3)

        hooks = add_fhmc_hooks(self.model.flow, self.model.target)
        sample_outputs = self.model.sample(sample_size)
        grid_outputs = self.model(
            (inputs, torch.full((grid_size**2,), -log(4 * π)))
        )
        [hook.remove() for hook in hooks]

        self._theta_grid = -(theta_grid - π / 2)  # different coordinates
        self._phi_grid = phi_grid

        self._grid_outputs = grid_outputs
        self._sample_outputs = sample_outputs

        self._grid_size = grid_size
        self._projection = projection

        """
        self._inputs = outputs["inputs"].detach()
        self._input_angles = sphere_vectors_to_angles(self._inputs)
        self._outputs = outputs["outputs"].detach()
        self._output_angles = sphere_vectors_to_angles(self._outputs)
        self._log_density = outputs["log_density"].detach()
        self._log_weights = outputs["log_weights"].detach()
        """

    def _make_heatmap(self, data: Tensor, title: str) -> Figure:
        fig, ax = plt.subplots(subplot_kw=dict(projection=self._projection))
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        cf = ax.pcolormesh(
            self._phi_grid,
            self._theta_grid,
            data.detach().view_as(self._phi_grid),
            cmap="viridis",
            shading="nearest",
        )
        fig.colorbar(cf, ax=ax, shrink=0.55)
        ax.set_title(title)
        fig.tight_layout()
        return fig

    def _make_scatter(self, data: Tensor, title: str) -> Figure:
        fig, ax = plt.subplots(subplot_kw=dict(projection=self._projection))
        ax.set_yticklabels([])
        ax.set_xticklabels([])


        coords = sphere_vectors_to_angles(
            self._sample_outputs["outputs"].detach()
        )
        lat, lon = coords.split(1, dim=-1)
        lon = lon.add(π).remainder(2 * π).sub(π)
        # lat = -(lat - π / 2)
        
        outlier_mask = data.abs() > (data.mean() + 3 * data.std())
        outliers_lon = lon[outlier_mask]
        outliers_lat = lat[outlier_mask]
        outliers_data = data[outlier_mask]
        
        cmap = ScalarMappable(
            norm=Normalize(vmin=data.min(), vmax=data.max()),
            cmap="viridis",
        )

        ax.scatter(lon, lat, s=3, c=cmap.to_rgba(data.detach()))

        ax.scatter(
            outliers_lon,
            outliers_lat,
            s=9,
            c=cmap.to_rgba(outliers_data.detach()),
            linewidths=1,
            edgecolors="black",
        )

        ax.set_title(title)
        fig.colorbar(cmap, ax=ax, shrink=0.55)
        fig.tight_layout()

        return fig

    def density_heatmap(self) -> Figure:
        data = self._grid_outputs["log_density"].exp()
        return self._make_heatmap(data, "Model density")

    def density_scatter(self) -> Figure:
        data = self._sample_outputs["log_density"].exp()
        return self._make_scatter(data, "Model density")

    def log_density_heatmap(self) -> Figure:
        data = self._grid_outputs["log_density"]
        return self._make_heatmap(data, "Model log density")

    def log_density_scatter(self) -> Figure:
        data = self._sample_outputs["log_density"]
        return self._make_scatter(data, "Model log density")

    def log_weights_heatmap(self) -> Figure:
        data = self._grid_outputs["log_weights"]
        return self._make_heatmap(data, "Log weights")

    def log_weights_scatter(self) -> Figure:
        data = self._sample_outputs["log_weights"]
        return self._make_scatter(data, "Log weights")

    def force_heatmap_x(self) -> Figure:
        data = self._grid_outputs["inputs"].grad[:, 0]
        return self._make_heatmap(data, "Force (x component)")

    def force_scatter_x(self) -> Figure:
        data = self._sample_outputs["inputs"].grad[:, 0]
        return self._make_scatter(data, "Force (x component)")

    def force_heatmap_y(self) -> Figure:
        data = self._grid_outputs["inputs"].grad[:, 1]
        return self._make_heatmap(data, "Force (y component)")

    def force_scatter_y(self) -> Figure:
        data = self._sample_outputs["inputs"].grad[:, 1]
        return self._make_scatter(data, "Force (y component)")

    def force_heatmap_z(self) -> Figure:
        data = self._grid_outputs["inputs"].grad[:, 2]
        return self._make_heatmap(data, "Force (z component)")

    def force_scatter_z(self) -> Figure:
        data = self._sample_outputs["inputs"].grad[:, 2]
        return self._make_scatter(data, "Force (z component)")

    def force_heatmap_abs(self) -> Figure:
        data = LA.vector_norm(self._grid_outputs["inputs"].grad, dim=-1)
        return self._make_heatmap(data, "Force (absolute)")

    def histogram(self) -> Figure:
        fig, ax = plt.subplots(subplot_kw=dict(projection=self._projection))
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        θ_in, ϕ_in = self._input_angles.split(1, dim=-1)
        θ_out, ϕ_out = self._output_angles.split(1, dim=-1)

        θ_in = (
            θ_in.add(π / 2).remainder(π).sub(π / 2).squeeze()
        )  # .view(self._grid_size, self._grid_size)
        θ_out = (
            θ_out.add(π / 2).remainder(π).sub(π / 2).squeeze()
        )  # .view(self._grid_size, self._grid_size)
        ϕ_in = (
            ϕ_in.add(π).remainder(2 * π).sub(π).squeeze()
        )  # .view(self._grid_size, self._grid_size)
        ϕ_out = (
            ϕ_out.add(π).remainder(2 * π).sub(π).squeeze()
        )  # .view(self._grid_size, self._grid_size)

        hist, bin_edges = torch.histogramdd(
            torch.stack(
                [θ_out, ϕ_out],
                dim=-1,
            ),
            bins=[θ_in, ϕ_in],
            density=True,
        )
        lat, lon = bin_edges[0], bin_edges[1]

        print(
            ϕ_in.isnan().any(),
            θ_in.isnan().any(),
            θ_out.isnan().any(),
            ϕ_out.isnan().any(),
        )
        print(ϕ_in.max(), ϕ_out.max())
        print(θ_in.max(), θ_out.max())
        print(hist)

        ax.pcolormesh(
            lon,
            lat,
            hist,
            cmap="viridis",
            shading="auto",
        )
        fig.tight_layout()
        return fig
        """hist, bin_edges = torch.histogramdd(
            torch.cat(
                [
                    θ_out.add(π / 2).remainder(π).sub(π / 2),
                    ϕ_out.add(π).remainder(2 * π).sub(π),
                ],
                dim=-1,
            ),
            bins=[θ_grid.add(π / 2).remainder(π).sub(π / 2)
                , ϕ_grid.add(π).remainder(2 * π).sub(π)
                ],
            density=True,
        )

        lat, lon = self._bin_edges[0], self._bin_edges[1]

        assert (lat >= -π / 2).all() and (lat <= π / 2).all()
        assert (lon >= -π).all() and (lon <= π).all()
        """


class RecursiveS2FlowVisualiser(FlowVisualiserBase):
    _figures: list[str] = [
        "histogram",
        "pairplot",
    ]

    def __init__(
        self,
        model: FlowBasedModel,
        sample_size: int = int(1e5),
        bins: int = 30,
        projection: str = "aitoff",
    ):
        assert model.target.dim == 2
        super().__init__(model, sample_size)

        self._input_angles = sphere_vectors_to_angles(self._inputs)
        self._output_angles = sphere_vectors_to_angles(self._outputs)

        θ_grid = torch.linspace(-π / 2, π / 2, bins + 1)
        ϕ_grid = torch.linspace(-π, π, bins + 1)
        θ_out, ϕ_out = self._output_angles.split(1, dim=-1)
        self._hist, self._bin_edges = torch.histogramdd(
            torch.cat(
                [
                    θ_out.add(π / 2).remainder(π).sub(π / 2),
                    ϕ_out.add(π).remainder(2 * π).sub(π),
                ],
                dim=-1,
            ),
            bins=[θ_grid, ϕ_grid],
            density=True,
        )

        self._projection = projection

    def histogram(self) -> Figure:
        fig, ax = plt.subplots(subplot_kw=dict(projection=self._projection))
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        lat, lon = self._bin_edges[0], self._bin_edges[1]

        assert (lat >= -π / 2).all() and (lat <= π / 2).all()
        assert (lon >= -π).all() and (lon <= π).all()

        ax.pcolormesh(
            lon,
            lat,
            self._hist,
            cmap="viridis",
            shading="auto",
        )
        fig.tight_layout()
        return fig


def histogram(
    xyz: Tensor,
    bins: int = 50,
    projection: str = "aitoff",
    **pcolormesh_kwargs,
) -> plt.Figure:
    x, y, z = xyz.flatten(end_dim=-2).split(1, dim=-1)  # merges all batches
    θϕ = torch.cat([torch.asin(z), torch.atan2(y, x)], dim=-1)

    θ_grid = torch.linspace(-π / 2, π / 2, bins + 1)
    ϕ_grid = torch.linspace(-π, π, bins + 1)

    hist, _ = torch.histogramdd(θϕ, bins=[θ_grid, ϕ_grid], density=True)

    fig, ax = plt.subplots(subplot_kw=dict(projection=projection))
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    pcolormesh_kwargs = (
        dict(cmap="viridis", shading="auto") | pcolormesh_kwargs
    )

    ax.pcolormesh(
        C=hist,
        X=ϕ_grid,
        Y=θ_grid,
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
    **scatter_kwargs,
) -> plt.Figure:
    x, y, z = xyz.split(1, dim=-1)
    θ = torch.asin(z)
    ϕ = torch.atan2(y, x)

    if colours is not None:
        cmap = ScalarMappable(
            norm=Normalize(vmin=colours.min(), vmax=colours.max()),
            cmap="viridis",
        )
        colours = cmap.to_rgba(colours)

    fig, ax = plt.subplots(subplot_kw=dict(projection=projection))
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    ax.scatter(ϕ, θ, c=colours, **scatter_kwargs)

    if colours is not None:
        fig.colorbar(cmap, ax=ax, shrink=1)

    fig.tight_layout()

    return fig


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


"""
def get_visualiser(flow: Flow) -> FlowVisualiserBase:

    if isinstance(flow, CircularFlow):
        return CircularFlowVisualiser
    elif isinstance(flow, RecursiveFlowS2):
        return RecursiveFlowS2Visualiser
"""
