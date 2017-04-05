import sys, glob, os
import numpy as np

from scipy.stats import norm
gaussian = norm

__all__ = ["Parameter", "ParameterSet"]


class Parameter(object):
    """Basically wraps a value with some methods that are useful for parameters,
    especially an associated Free/Fixed property.
    """
    name = 'template'
    _value = 0.0

    def __init__(self, name, initial=0.0, prior=None, free=True):
        self.name = name
        self.free = free
        if prior is not None:
            self._prior = prior

        self.value = initial
            
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

    @property
    def free_names(self):
        if self.free:
            return [self.name]
        else:
            return []
            
    @property
    def params(self):
        return dict([(self.name, self.value)])

    def prior_prob(self, **kwargs):
        try:
            return self._prior(value=self.value, **kwargs)
        except(AttributeError):
            return 1.0


class ParameterSet(Parameter):
    """Container for a set of parameters.  Try to be recursive.
    """
    name = 'Test'

    def __init__(self, paramlist=[], name='Test'):
        self._paramlist = paramlist
        #self.params = dict(zip(self.parnames, paramlist)
        self.name = name

    @property
    def free(self):
        f = [p.free for p in self._paramlist]
        return True in f

    @property
    def params(self):
        return dict([(p.name, p) for p in self._paramlist])

    def __repr__(self):
        return '{}({})'.format(self.__class__, self.name)

    def update(self, **params):
        """Update named parameters based on supplied keyword arguments.
        Needs to be fast.
        """
        #tpnames = self.parnames
        for p, v in params.items():
            self.params[p].value = v

    def __getitem__(self, k):
        return self.params[k].value

    def __setitem__(self, k, v):
        self.params[k].value = v

    @property
    def value(self):
        """A vector of the current values for the free parameters, including
        lower-level parameters by recursion.
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
        """A list of just the free parameters (as objects).
        """
        return [p for p in self._paramlist if p.free]

    @property
    def free_names(self):
        """A list of the names of the free parameters, including lower-level
        parameters by recursion.
        """
        n = []
        for p in self.free_params:
            n += p.theta_names
        return n

    @property
    def parnames(self):
        """A list of the name of the parameters in this set.  Does not move
        recursively down into parameters.
        """
        return [p.name for p in self._paramlist]

    @property
    def npar(self):
        return len(self.free_params)

    @property
    def ndim(self):
        return len(self.value)
