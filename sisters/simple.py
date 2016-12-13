import sys, glob, os
import numpy as np

from emcee import EnsembleSampler
from scipy.stats import norm as gaussian

#gaussian = norm()


class Parameter(object):
    """Basically wraps a value with some methods that are useful for parameters,
    especially an associated Free/Fixed property.
    """
    
    name = 'template'
    _value = 0.0
    free = True
    
    def __init__(self, name, initial=0.0, free=None):
        self.name = name
        self.value = initial
        if free is not None:
            self.free = free

    def __repr__(self):
        return '{}({})'.format(self.__class__, self.name)

    def __len__(self):
        try:
            return len(self.value)
        except(TypeError):
            return 1

    @property
    def value(self):
        return np.atleast_1d(self._value)

    @value.setter
    def value(self, v):
        self._value = np.atleast_1d(v)
        if not self.free:
            print("Warning: setting fixed parameter {}".format(self.name))


class ParameterSet(Parameter):
    """Container for a set of parameters.
    """

    name='Test'
    
    def __init__(self, paramlist=[], name='Test'):
        self.params = paramlist
        self.name = name

    def __repr__(self):
        return '{}({})'.format(self.__class__, self.name)

    def make_fixed(self, pname):
        self.params[self.parnames.index(pname)].free = False

    def make_free(self, pname):
        self.params[self.parnames.index(pname)].free = True

    def remove(self, pname):
        pass
        
    def update(self, **params):
        """Update named parameters based on supplied keyword arguments.
        Needs to be fast.
        """
        tpnames = self.parnames
        for p, v in params.items():
            self.params[tpnames.index(p)].value = v

    def __getitem__(self, k):
        return self.params[self.parnames.index(k)].value

    @property
    def value(self):
        """A vector of the current values for the free parameters.
        """
        return np.concatenate([p.value for p in self.free_params])

    @value.setter
    def value(self, vec):
        """Set the parameter values using a vector.
        """
        start = 0
        assert len(vec) == self.ndim
        for i, p in enumerate(self.free_params):
            stop = start + len(p)
            p.value = vec[start:stop]
            start = stop

    @property
    def free_params(self):
        """A list of just the free parameters.
        """
        return [p for p in self.params if p.free]

    @property
    def theta_names(self):
        """A list of the names of the free parameters.
        """
        return [p.name for p in self.free_params]

    @property
    def parnames(self):
        return [p.name for p in self.params]


    @property
    def npar(self):
        return len(self.free_params)
    
    @property
    def ndim(self):
        return len(self.value)


class GaussianPriorND(ParameterSet):

    params = []

    def __init__(self, dimensions=['age', 'distance', 'feh'], name='Cluster'):
        self.name = name
        self.params = []
        self.dimensions = dimensions
        for d in self.dimensions:
            self.params += [Parameter('{}_mean'.format(d)),
                            Parameter('{}_sigma'.format(d))]
        
    def likelihood(self, theta, samples):
        #print(theta)
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


def lnpostfn(theta, samples=[]):
    print(theta)
    lnp = np.sum([np.log(model.likelihood(theta, s)) for s in samples])
    lnp += -np.log(theta[1])
    if np.any(theta < 1e-8) or (np.isfinite(lnp) is False):
        return -np.inf
    else:
        return lnp


def mock_samples(s, N, precision=10, **extras):
    """For a given set of stellar parameters and 
    """
    dims = s.dtype.names
    dt = np.dtype([(d, np.float) for d in dims])
    chain = np.zeros(N, dtype=dt)
    for d in dims:
        chain[d] = gaussian.rvs(loc=s[d], scale=s[d] / precision, size=N)
    return chain
    

def simulate(model, Nstar, Nsample, **extras):
    """Simulate a set of true values from the distribution
    """
    stars = model.draw(Nstar)
    samples = [mock_samples(s, Nsample, **extras) for i,s in enumerate(stars)]

    return samples, stars


if __name__ == "__main__":

    rp = {'precision': 10.0,
          'nwalkers': 128,
          'niter': 256}

    model = GaussianPriorND(dimensions=['age'])
    model.value = [0.2, 0.001]
    model.initial_value = model.value.copy()

    Nstar = 50
    Nsamples = 1000
    mock_data, stars = simulate(model, Nstar, Nsamples, **rp)

    #sys.exit()
    
    postkwargs = {'samples': mock_data}
    esampler = EnsembleSampler(rp['nwalkers'], model.ndim, lnpostfn,
                               kwargs=postkwargs)

    initial = [0.1, 0.01]
    initial = [np.random.normal(loc=i, scale=0.1 * i, size=(rp['nwalkers']))
               for i in initial]
    initial = np.array(initial).T
    
    for i, result in enumerate(esampler.sample(initial, iterations=rp['niter'],
                                               storechain=True)):

            if (i % 10) == 0:
                print(i)
