Just a scratch pad to investigate flow-based sampling on spheres, using the simple von Mises-Fisher family of distributions as the target.

To do:
- [x] Combine unconditional and conditional flows into single class
- [ ] Add target density that is a mixture of von Mises densities
- [ ] Implement [exponential map flows](https://arxiv.org/abs/2002.02428)
- [ ] Implement [bump layers](https://arxiv.org/abs/2110.00351)
- [ ] Check that both models and sampling works when $D > 2$
- [ ] Tidy up, make variable names consistent etc.
- [ ] Integrate working models into [torchnf](https://github.com/marshrossney/torchnf) package
