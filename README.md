Just a scratch pad to investigate flow-based sampling on spheres, using the simple von Mises-Fisher family of distributions as the target.

To do:
- [x] Combine unconditional and conditional flows into single class
- [x] Add target density that is a mixture of von Mises densities
- [x] Add progress bars to HMC simulations
- [x] Scripts to train models and run Flow-HMC algorithm
- [ ] Implement [exponential map flows](https://arxiv.org/abs/2002.02428)
- [ ] Implement [bump layers](https://arxiv.org/abs/2110.00351)
- [ ] Extend models, sampling, and some visualisation to $D > 2$
- [ ] Add documentation and references
- [ ] Add installation and usage instructions and example outputs in this README
- [ ] Integrate working models into [torchnf](https://github.com/marshrossney/torchnf) package
