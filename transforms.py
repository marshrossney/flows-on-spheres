from itertools import chain
import math

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

Tensor = torch.Tensor

PI = math.pi


class MobiusTransform:
    """
    Mobius transform from the circle to itself.

        -y = (1 - |ω|^2) / |x - ω|^2 * (x - ω) - ω
        
        |dy/dx| = (1 - |ω|^2) / |x - ω|^2
    
    The transformation takes a batch of inputs `x` with shape
    (n_batch, ..., 2) that are (x, y) coordinates on the unit circle,
    and a batch of parameters `omega` with the same shape, but the
    (x, y) coordinates live on the unit disk. The transformation
    returns coordinates `y` on the unit circle.
    """
    
    @property
    def identity_params(self) -> Tensor:
        return torch.tensor([0.0, 0.0])
    
    @property
    def n_params(self) -> int:
        return 2
    
    def forward(self, x: Tensor, omega: Tensor) -> tuple[Tensor, Tensor]:
        assert x.shape[-1] == 2
        assert x.shape == omega.shape
        #assert torch.allclose(x.pow(2).sum(dim=-1), torch.ones(x.shape[:-1]))
        assert torch.all(omega.pow(2).sum(dim=-1) < 1), f"{omega.pow(2).sum(dim=-1).abs().max()}"
        
        coeff = (1 - omega.pow(2).sum(dim=-1, keepdim=True)) / (x - omega).pow(2).sum(dim=-1, keepdim=True)
        y = - (coeff * (x - omega) - omega)
        ldj = torch.log(coeff).flatten(start_dim=1).sum(dim=1)
        
        return y, ldj
    
    def __call__(self, x, params):
        return self.forward(x, params)
        


class RQSplineTransform:
    """
    Pointwise rational quadratic spline transformation.
    """

    def __init__(self, interval: tuple[float], n_segments: int) -> None:
        self.interval = interval
        self.n_segments = n_segments

    @property
    def n_knots(self) -> int:
        return self.n_segments + 1

    @property
    def n_params(self) -> int:
        return 2 * self.n_segments + self.n_knots

    @property
    def identity_params(self) -> Tensor:
        return torch.cat(
            (
                torch.full(
                    size=(2 * self.n_segments,),
                    fill_value=1 / self.n_segments,
                ),
                (torch.ones(self.n_knots).exp() - 1).log(),
            ),
            dim=0,
        )

    @staticmethod
    def pad_derivs(derivs: Tensor) -> Tensor:
        return derivs

    def build_spline(
        self,
        widths: Tensor,
        heights: Tensor,
        derivs: Tensor,
    ) -> tuple[Tensor]:
        """Builds a rational quadratic spline function."""
        # Normalise the widths and heights to the interval
        interval_size = self.interval[1] - self.interval[0]
        widths = torch.nn.functional.softmax(widths, dim=-1).mul(interval_size)
        heights = torch.nn.functional.softmax(heights, dim=-1).mul(interval_size)

        # Let the derivatives be positive definite
        derivs = torch.nn.functional.softplus(derivs)
        derivs = self.pad_derivs(derivs)

        # Just a convenient way to ensure it's on the correct device
        zeros = torch.zeros_like(widths).sum(dim=-1, keepdim=True)

        knots_xcoords = torch.cat(
            (
                zeros,
                torch.cumsum(widths, dim=-1),
            ),
            dim=-1,
        ).add(self.interval[0])

        knots_ycoords = torch.cat(
            (
                zeros,
                torch.cumsum(heights, dim=-1),
            ),
            dim=-1,
        ).add(self.interval[0])

        return widths, heights, derivs, knots_xcoords, knots_ycoords

    def forward(self, x: Tensor, params: Tensor) -> tuple[Tensor, Tensor]:
        x = x.contiguous()
        # params = params.unsqueeze(dim=1)  # shape (batch, 1, n_params)

        widths, heights, derivs = params.split(
            (self.n_segments, self.n_segments, self.n_knots),
            dim=-1,
        )
        (
            widths,
            heights,
            derivs,
            knots_xcoords,
            knots_ycoords,
        ) = self.build_spline(widths, heights, derivs)

        segment_idx = torch.searchsorted(knots_xcoords, x.unsqueeze(-1)) - 1
        segment_idx.clamp_(0, self.n_segments - 1)

        # Get parameters of the segments that x falls in
        w = torch.gather(widths, -1, segment_idx).squeeze(-1)
        h = torch.gather(heights, -1, segment_idx).squeeze(-1)
        d0 = torch.gather(derivs, -1, segment_idx).squeeze(-1)
        d1 = torch.gather(derivs, -1, segment_idx + 1).squeeze(-1)
        x0 = torch.gather(knots_xcoords, -1, segment_idx).squeeze(-1)
        y0 = torch.gather(knots_ycoords, -1, segment_idx).squeeze(-1)

        s = h / w
        s.clamp(1e-4, 1 - 1e-4)  # don't want tiny slopes

        alpha = (x - x0) / w
        alpha.clamp_(1e-4, 1 - 1e-4)

        denominator_recip = torch.reciprocal(
            s + (d1 + d0 - 2 * s) * alpha * (1 - alpha)
        )

        beta = (s * alpha.pow(2) + d0 * alpha * (1 - alpha)) * denominator_recip
        y = y0 + h * beta

        gradient = (
            s.pow(2)
            * (
                d1 * alpha.pow(2)
                + 2 * s * alpha * (1 - alpha)
                + d0 * (1 - alpha).pow(2)
            )
            * denominator_recip.pow(2)
        )

        if gradient.less(0.0).any():
            log.warning("Gradient has negative values")
            gradient = gradient.clamp(min=1e-6)

        ldj = gradient.log().flatten(start_dim=1).sum(dim=1)

        return y, ldj

    def __call__(self, x: Tensor, params: Tensor) -> tuple[Tensor, Tensor]:
        return self.forward(x, params)

    def visualise(self, params: Tensor) -> plt.Figure:
        assert params.dim() == 2


class RQSplineTransformCircularDomain(RQSplineTransform):
    """
    Rational quadratic spline transformation on the circle.
    """

    def __init__(self, n_segments: int):
        super().__init__((0, 2 * PI), n_segments)

    @property
    def n_knots(self) -> int:
        return self.n_segments

    @staticmethod
    def pad_derivs(derivs: Tensor) -> Tensor:
        return F.pad(derivs.flatten(1, -2), (0, 1), "circular").view(
            *derivs.shape[:-1], -1
        )

class C2SplineTransform:
    
    def __init__(self, interval: tuple[float], n_segments: int) -> None:
        #self.a, self.b = interval
        self.interval = interval
        self.n_segments = n_segments
    
    @property
    def identity_params(self) -> Tensor:
        return torch.ones(self.n_segments * 2 + 2)
    
    def build_spline(
        self,
        intervals: Tensor,
        weights: Tensor,
    ) -> tuple[Tensor]:
        
        #Δ = F.softplus(intervals) + 1e-3  # 
        Δ = torch.sigmoid(intervals) + 0.1
        Δ = Δ / Δ.sum(dim=-1, keepdim=True)
        Δpad = F.pad(Δ, (1, 1), "constant", 0)
        
        #print(f"Δ = {Δ.shape}")
        
        ρ = F.softplus(weights) + 1e-3  # min weight TODO make configurable
        ρ = ρ / (((ρ[..., :-2] + ρ[..., 1:-1] + ρ[..., 2:]) / 3) * Δ).sum(dim=-1, keepdim=True)
        
        ω = (ρ[..., 1:] - ρ[..., :-1]) / (Δpad[..., :-1] + Δpad[..., 1:])
        I = ρ[..., 1:-1] * Δ + (1/3) * (ω[..., 1:] - ω[..., :-1]) * Δ ** 2
        
        #print(f"ρ = {ρ.shape}")
        #print(f"ω = {ω.shape}")

        #Y = np.concatenate(([a,], a + np.cumsum(I, dim=-1)))

        zeros = torch.zeros_like(I).sum(dim=-1, keepdim=True)

        X = torch.cat(
            (
                zeros,
                torch.cumsum(Δ, dim=-1),
            ),
            dim=-1,
        )
        Y = torch.cat(
            (
                zeros,
                torch.cumsum(I, dim=-1),
            ),
            dim=-1,
        )
        #print(f"X = {X.shape}")
        #print(f"Y = {Y.shape}")
        
        return Δpad, ρ, ω, X, Y
    
    def forward(self, x: Tensor, params: Tensor) -> tuple[Tensor, Tensor]:
        x = x.contiguous()
        #params = params.unsqueeze(dim=1)  # shape (batch, 1, n_params)
        
        #print("x: ", x.min(), x.max())
        x = x - self.interval[0]
        x = x / (self.interval[1] - self.interval[0])
        
        intervals, weights = params.split(
            (self.n_segments, self.n_segments + 2),
            dim=-1,
        )
        Δ, ρ, ω, X, Y = self.build_spline(intervals, weights)
        #print(f"Δ = {Δ[0].tolist()},  ρ = {ρ[0].tolist()}")
        
        #print(f"x = {x.shape}")


        ix = torch.searchsorted(X, x.unsqueeze(-1), side="right")
        #print(ix.min(), ix.max())
        ix.clamp_(1, self.n_segments)

        # Get parameters of the segments that x falls in
        Δ = torch.gather(Δ, -1, ix).squeeze(-1)
        ρ = torch.gather(ρ, -1, ix).squeeze(-1)
        ωi = torch.gather(ω, -1, ix).squeeze(-1)
        ωim1 = torch.gather(ω, -1, ix - 1).squeeze(-1)
        x0 = torch.gather(X, -1, ix - 1).squeeze(-1)
        y0 = torch.gather(Y, -1, ix - 1).squeeze(-1)
        
        #if not torch.all(x > x0):
        #    print(f"x = {x[~(x > x0)]}, x0 = {x0[~(x > x0)]}")
        #if not torch.all(x < x0 + Δ):
        #    print(f"x = {x[~(x < x0 + Δ)]}, x0 + Δ = {(x0 + Δ)[~(x < x0 + Δ)]}")

        θ = (x - x0) / Δ
        
        #if not torch.all(θ >= 0):
        #    print(f"θ = {θ[~(θ >= 0)]}")
        #if not torch.all(θ < 1):
        #    print(f"θ = {θ[~(θ < 1)]}")

        #alpha.clamp_(1e-4, 1 - 1e-4)
        
        y = (
            y0
            + ρ * Δ * θ
            - ωim1 * Δ ** 2 * θ * (1 - θ)
            + (1/3) * (ωi - ωim1) * Δ ** 2 * θ ** 3
        )
        
        #print("X: ", X[0])
        #print("Y: ", Y[0])
        #print("x: ", x.min(), x.max())
        #print("y: ", y.min(), y.max())
        gradient = (
            ρ
            + ωi * Δ * θ ** 2
            - ωim1 * Δ * (1 - θ) ** 2
        )
        
        if gradient.less(0.0).any():
            values = gradient[gradient < 0]
            print(f"Gradient has negative values: {values.tolist()}")
            print(f"interval is: ix = {ix[gradient < 0]}")
            print(f"theta is: {θ[gradient < 0]}")
            print(f"Delta is: {Δ[gradient < 0]}")
            print(f"rho is: {ρ[gradient < 0]}")
            print(f"omega is: {ωi[gradient < 0]}")
            print(f"omega-1 is: {ωim1[gradient < 0]}")
            gradient = gradient.clamp(min=1e-6)

        ldj = gradient.log().flatten(start_dim=1).sum(dim=1)
        
        y = y * (self.interval[1] - self.interval[0])
        y = y + self.interval[0]
        ldj += math.log(self.interval[1] - self.interval[0]) * torch.numel(y[0])
        
        #print("y: ", y.min(), y.max())

        return y, ldj

    def __call__(self, x: Tensor, params: Tensor) -> tuple[Tensor, Tensor]:
        return self.forward(x, params)


