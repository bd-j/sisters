kith
=====

Given samples from N distributions p<sub>n</sub>(t, m, z, d, a) corresponding to N stars that are members of a cluster, infer parameters of the hyper-prior p(T, Z, D) that describes the cluster age, metallicity, distance, and intrinsic distributions thereof.  This involves a marginalization over mass(m) and attenuation(a).

In principle this method allows for heterogenous constraints on different stars, individual cluster membership probabilities, etc.

To do:
  - [ ] Output chains for all parameters of the member stars incorporating the new hyper-prior.

  - [ ] Incorporate a prior IMF p(m | t) ~ m<sup>-&Gamma;</sup> when marginalizing over mass(m), with the age-dependence coming from the maximum mass for a given age.

  - [ ] Outlier modeling.

  - [ ] Extend to more complicated SFHs than SSPs/Gaussians

  - [ ] Incorporate selection effects.

  - [ ] Figure out what to do about binaries when the individual stellar chains include the mass-ratio(r).

  - [ ] Compute derivatives (with theano or autodiff) for faster optimization and sampling.
