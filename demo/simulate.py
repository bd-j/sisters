import sys, os, glob
import numpy as np

from scipy.stats import norm, genlogistic, logistic, genhalflogistic

from kith.model import GaussianPriorND, lnpostfn
from emcee import EnsembleSampler

gaussian = norm


def mock_samples(s, N, precision=10, **extras):
    """For a given set of stellar parameters and precision, mock up samples
    from a gaussian.
    """
    dims = s.dtype.names
    dt = np.dtype([(d, np.float) for d in dims])
    chain = np.zeros(N, dtype=dt)
    for d in dims:
        chain[d] = gaussian.rvs(loc=s[d], scale=s[d] / precision, size=N)
    return chain


def mock_samples_sigmoid(s, N, upper=False, scale=1.0,
                         maxv={'age': 13.7, 'feh': 0.5, 'distance': 100},
                         **extras):
    """For a given set of stellar parameters and precision, mock up samples
    from a logistic function.
    """
    dims = s.dtype.names
    dt = np.dtype([(d, np.float) for d in dims])
    chain = np.zeros(N, dtype=dt)

    for d in dims:
        loc = np.random.uniform(0, s[d])
        # if upper:
        #     loc = np.random.uniform(0, s[d])
        # else:
        #     loc = np.random.uniform(s[d], maxv[d])
    return chain


def simulate(model, Nstar, Nsample, **extras):
    """Simulate a set of true values from the distribution
    """
    stars = model.draw(Nstar)
    samples = [mock_samples(s, Nsample, **extras) for i, s in enumerate(stars)]

    return samples, stars


if __name__ == "__main__":

    rp = {'precision': 10.0,
          'nwalkers': 64,
          'niter': 256}

    model = GaussianPriorND(dimensions=['age', 'distance'])
    model.update(age_mean=0.2, age_sigma=0.001,
                 distance_mean=400, distance_sigma=10)
    model.initial_value = model.value.copy()

    Nstar = 50
    Nsamples = 1000
    mock_data, stars = simulate(model, Nstar, Nsamples, **rp)

    postkwargs = {'samples': mock_data, 'model': model}
    esampler = EnsembleSampler(rp['nwalkers'], model.ndim, lnpostfn,
                               kwargs=postkwargs)

    initial = [0.1, 0.01, 500, 100]
    initial = [np.random.normal(loc=i, scale=0.1 * i, size=(rp['nwalkers']))
               for i in initial]
    initial = np.array(initial).T

    for i, result in enumerate(esampler.sample(initial,
                                               iterations=rp['niter'],
                                               storechain=True)):

            if (i % 10) == 0:
                print(i)
