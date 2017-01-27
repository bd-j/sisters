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

    @property
    def prior_prob(self):
        try:
            return self.prior(self.value, **self.prior_kwargs)
        except(AttributeError):
            return 1.0


class ParameterSet(Parameter):
    """Container for a set of parameters.
    """
    name = 'Test'

    def __init__(self, paramlist=[], name='Test'):
        self._paramlist = paramlist
        #self.params = dict(zip(self.parnames, paramlist)
        self.name = name

    @property
    def params(self):
        return dict([(p.name, p) for p in self._paramlist])

    def __repr__(self):
        return '{}({})'.format(self.__class__, self.name)

    def make_fixed(self, pname):
        self.params[pname].free = False

    def make_free(self, pname):
        self.params[pname].free = True

    def remove(self, pname):
        pass

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
        return [p for p in self._paramlist if p.free]

    @property
    def theta_names(self):
        """A list of the names of the free parameters.
        """
        return [p.name for p in self.free_params]

    @property
    def parnames(self):
        return [p.name for p in self._paramlist]

    @property
    def npar(self):
        return len(self.free_params)

    @property
    def ndim(self):
        return len(self.value)
