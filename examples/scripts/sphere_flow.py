import matplotlib.pyplot as plt
import matplotlib

from vonmises.distributions import VonMisesFisherDensity, VonMisesFisherMixtureDensity
from vonmises.flows import RecursiveFlowS2
from vonmises.model import FlowBasedModel
from vonmises.transforms import MobiusMixtureTransform, BSplineTransform
from vonmises.utils import get_trainer
from vonmises.viz import SphericalFlowVisualiser, RecursiveS2FlowVisualiser


xy_transformer = MobiusMixtureTransform(10)
z_transformer = BSplineTransform(10)
target = VonMisesFisherMixtureDensity(
        κ=[10, 10, 10],
        μ=[(0, -1, -1), (0, 1, -1), (1, 0, 1)],
)
#target = VonMisesFisherDensity(κ=10, μ=(0, 0, 1))
flow = RecursiveFlowS2(z_transformer, xy_transformer, n_layers=1, net_hidden_shape=[])
model = FlowBasedModel(flow, target, batch_size=4000)
trainer = get_trainer(3000)

trainer.fit(model)
metrics, = trainer.test(model)

vis = SphericalFlowVisualiser(model)
#vis.histogram().savefig("hist.png")
vis.density_heatmap().savefig("density_heatmap.png", bbox_inches="tight")
vis.density_scatter().savefig("density_scatter.png", bbox_inches="tight")
#vis.log_weights_heatmap().savefig("log_weights_heatmap.png", bbox_inches="tight")
#vis.log_weights_scatter().savefig("log_weights_scatter.png", bbox_inches="tight")

vis.force_scatter_x().savefig("xforce.png", bbox_inches="tight")
vis.force_scatter_y().savefig("yforce.png", bbox_inches="tight")
vis.force_scatter_z().savefig("zforce.png", bbox_inches="tight")

#vis = RecursiveS2FlowVisualiser(model)

#for name, figure in vis.figures():
#    figure.savefig(f"{name}.png")

