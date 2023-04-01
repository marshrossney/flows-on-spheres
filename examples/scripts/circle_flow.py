import matplotlib.pyplot as plt
import matplotlib

from vonmises.distributions import VonMisesFisherDensity, VonMisesFisherMixtureDensity
from vonmises.flows import CircularFlow
from vonmises.model import FlowBasedModel
from vonmises.transforms import MobiusMixtureTransform, CircularRQSplineTransform
from vonmises.utils import get_trainer
from vonmises.viz import CircularFlowVisualiser


transformer = MobiusMixtureTransform(10)#CircularRQSplineTransform(10)
#target = VonMisesFisherMixtureDensity(
        #κ=[5, 10, 15],
        #μ=[(0, 1), (1, 0), (-1, 1)],
        #)
target = VonMisesFisherDensity(κ=10, μ=(-1, 1))
flow = CircularFlow(transformer, n_layers=1)
model = FlowBasedModel(flow, target, batch_size=4000)
trainer = get_trainer(4000)

trainer.fit(model)
metrics, = trainer.test(model)

vis = CircularFlowVisualiser(model)

for name, figure in vis.figures():
    figure.savefig(f"{name}.png")

