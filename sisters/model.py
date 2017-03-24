import sys, glob, os
import numpy as np

from scipy.stats import norm as gaussian

from .parameters import Parameter, ParameterSet


class GaussianPriorND(ParameterSet):

    #params = []

    def __init__(self, dimensions=['age', 'distance', 'feh'], name='Cluster'):
        self.name = name
        self._paramlist = []
        self.dimensions = dimensions
        for d in self.dimensions:
            self._paramlist += [Parameter('{}_mean'.format(d)),
                                Parameter('{}_sigma'.format(d))]

    def likelihood(self, theta, samples):
        """Calculate likelihood for a single chain (given by `samples`) at the
        value of the hyperparameters given by `theta`.  This is accomplished by
        summing the probabilities of each postion in the chain, given the
        hyperparameters `theta`.

        :param theta:
            The value of the hyperparameters.  Should be same length as
            `free_params`.

        :param samples:
            A dictionary or structured array giving the chain.  The keys (or
            field names) should include the contents of self.dimensions

        """
        # print(theta)
        self.value = theta
        return np.sum(self.__call__(samples))

    def __call__(self, samples):
        """A slow way to call a multivariate normal.  But points the way to
        evaluating more complicated joint (or separable) priors.
        """
        for d in self.dimensions:
            mu = self['{}_mean'.format(d)]
            sigma = self['{}_sigma'.format(d)]
            l = gaussian.pdf(samples[d], loc=mu, scale=sigma)
            try:
                like *= l
            except(NameError):
                like = l

        return like

    def draw(self, N):
        dt = np.dtype([(d, np.float) for d in self.dimensions])
        draws = np.zeros(N, dtype=dt)
        for d in self.dimensions:
            mu = self['{}_mean'.format(d)]
            sigma = self['{}_sigma'.format(d)]
            rv = gaussian(loc=mu, scale=sigma)
            draws[d] = rv.rvs(size=N)
        return draws


def lnpostfn(theta, samples=[], model=None):
    """A simple posterior probability function that sums the log-likelihood of
    each chain given the hyper-parameters theta, and applies any hyper-priors
    to these hyper-parameters.
    """
    if model is None:
        model = model
    print(theta)
    # Sum the log-likelihoods of each chain.
    lnp = np.sum([np.log(model.likelihood(theta, s)) for s in samples])
    # Prior on scale hyper-parameter.
    lnp += -np.log(model['age_sigma']) - np.log(model['dist_sigma'])
    if np.any(theta < 1e-8) or (np.isfinite(lnp) is False):
        return -np.inf
    else:
        return lnp
