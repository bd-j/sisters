import numpy as np
import scipy.stats


class Prior(object):
    
    def __init__(self, parnames=[], argnames=[], name='', **kwargs):
        """
        :param parnames:
            A list of names of the parameters params, used to alias the intrinsic
            parameter names.  This way different instances of the same Prior
            can (must) have different parameter names.
        """
        if len(parnames) == 0:
            parnames = self.prior_params
        assert len(parnames) == len(self.prior_params)
        self.alias = dict(zip(self.prior_params, parnames))
        self.params = {}
        if len(argnames) == 0:
            argnames = self.argnames
        self.argnames = argnames

        self.name = name

    def update(self, **kwargs):
        for k in self.prior_params:
            self.params[k] = kwargs[self.alias[k]]
        
        self.args = [kwargs[k] for k in self.argnames]
        self.x = self.args[0] # because univariate

    def __call__(self, **kwargs):
        self.update(**kwargs)
        return self.evaluate()

    def draw(self, **kwargs):
        pass


class GaussianPrior(Prior):

    argnames = ['value']
    prior_params = ['loc', 'scale']
    distribution = scipy.stats.norm

    def evaluate(self, **extras):
        return self.distribution.pdf(self.x, **self.params)


class PowerLawPrior(Prior):

    argnames = ['value']
    prior_params = ['slope', 'loc', 'scale'] # need this to be 'min', 'max' instead of scale, loc
    distribution = scipy.stats.powerlaw

    def evaluate(self, **extras):
        slope = self.params.pop('slope')
        return self.distribution.pdf(self.x, slope, **self.params)
    

class UniformPrior(Prior):

    argnames = ['value']
    prior_params = ['min', 'max']
    distribution = scipy.stats.uniform

    def evaluate(self, **extras):
        loc = self.params.pop('min')
        scale = self.params.pop('max') - loc
        return self.distribution.pdf(self.x, loc=loc, scale=scale)
