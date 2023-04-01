from typing import TypeAlias

import pytorch_lightning as pl
import torch
from jsonargparse.typing import PositiveInt, PositiveFloat

from vonmises.flows import Flow
from vonmises.distributions import Density, uniform_prior
from vonmises.metrics import metropolis_acceptance, effective_sample_size

Tensor: TypeAlias = torch.Tensor


class FlowBasedModel(pl.LightningModule):
    def __init__(
        self,
        flow: Flow,
        target: Density,
        batch_size: PositiveInt,
        test_sample_size: PositiveInt = pow(2, 14),
        init_lr: PositiveFloat = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.flow = flow
        self.target = target
        self.batch_size = batch_size
        self.test_sample_size = test_sample_size
        self.init_lr = init_lr

    def forward(self, inputs: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        x_in, log_prior_density = inputs
        x_out, delta_log_vol = self.flow(x_in)
        log_model_density = log_prior_density - delta_log_vol
        log_target_density = self.target.log_density(x_out)
        return dict(
            inputs=x_in,
            outputs=x_out,
            log_density=log_model_density,
            log_weights=log_target_density - log_model_density,
        )

    def training_step(self, inputs: tuple[Tensor, Tensor], *_) -> Tensor:
        outputs = self(inputs)
        log_weights = outputs["log_weights"]

        kl_div = log_weights.mean().negative()
        self.log("loss", kl_div)

        return kl_div

    def validation_step(self, inputs: tuple[Tensor, Tensor], *_) -> None:
        outputs = self(inputs)
        log_weights = outputs["log_weights"]

        self.log("val/acceptance", metropolis_acceptance(log_weights))
        self.log("val/ess", effective_sample_size(log_weights))
        self.log("val/kl_div", log_weights.mean().negative())

    def test_step(self, inputs: tuple[Tensor, Tensor], *_) -> None:
        outputs = self(inputs)
        log_weights = outputs["log_weights"]

        self.log("acceptance", metropolis_acceptance(log_weights))
        self.log("ess", effective_sample_size(log_weights))
        self.log("kl_div", log_weights.mean().negative())

    def sample(self, sample_size: int = 1) -> dict[str, Tensor]:
        prior = uniform_prior(self.target.dim, sample_size)
        return self(next(prior))

    def train_dataloader(self):
        return uniform_prior(self.target.dim, self.batch_size)

    def val_dataloader(self):
        return self.test_dataloader()

    def test_dataloader(self):
        return uniform_prior(self.target.dim, self.test_sample_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.init_lr)
