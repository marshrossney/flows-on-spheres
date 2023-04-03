import matplotlib.pyplot as plt
import matplotlib

from vonmises.distributions import VonMisesFisherDensity, VonMisesFisherMixtureDensity
from vonmises.flows import RecursiveFlowS2
from vonmises.model import FlowBasedModel
from vonmises.transforms import MobiusMixtureTransform, BSplineTransform
from vonmises.utils import get_trainer
from vonmises.viz import SphericalFlowVisualiser


xy_transformer = MobiusMixtureTransform(10)
z_transformer = BSplineTransform(10)
#target = VonMisesFisherMixtureDensity(
#        κ=[10, 10, 10],
#        μ=[(0, -1, -1), (0, 1, -1), (1, 0, 1)],
#)
target = VonMisesFisherDensity(κ=10, μ=(1, 0, 0))
flow = RecursiveFlowS2(z_transformer, xy_transformer, n_layers=1, net_hidden_shape=[])
model = FlowBasedModel(flow, target, batch_size=4000)
trainer = get_trainer(3000)

trainer.fit(model)
metrics, = trainer.test(model)

vis = SphericalFlowVisualiser(model)


for name, fig in vis.figures(coords="outputs", topk=1000):
    fig.savefig(name + ".png", bbox_inches="tight")

