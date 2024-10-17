import numpy as np
from sisters.parameters import Parameter, ParameterSet


class Likelihood(ParameterSet):

    def lnp_of_samples(self, chain):
        """Get the log-prior-probability for samples from the chain.  This is given
        by the prior probability for the nominally fixed parameters that part
        of the chain.

        :param chain:
            A dict or structured array with keys (or fields) that give the name
            of each of the parameters of the chain, and values that are the
            value of the chain for that parameter at each iteration.

        :returns prob:
            The product of the probabilities for each parameter in the
            chain. ndarray of same length as the chain.
        """
        for d in self.dimensions:
            p = self.params[d].prior_prob(value=chain[d], **self.params)
            try:
                prob += np.log(p)
            except(NameError):
                prob = np.log(p)

    @property
    def lnp_prior(self):
        """Calculate the log-prior-probability of the free parameters.  This assumes
        that the value of the parameters has already been set
        """
        lnp = np.sum([np.log(par.prior_prob(**self.params))
                      for par in self.free_params])
        return lnp
        
    #def lnp_of_samples(self, chain):
    #    return self.lnp_prior + self.sample_lnp(chain)

    def integrate_chain(self, chain):
        lnprob_chain = np.logaddexp(self.lnp_of_samples(chain))
        return lnprob_chain

    def likelihood(self, theta, chains):
        self.value = theta
        lnp = 0.
        for chain in chains:            
            lnp_chain = self.integrate_chain(chain)
            lnp += lnp_chain

        return lnp_chain + self.lnp_prior


class MixtureLikelihood(ParameterSet):

    def likelihood(self, theta, chains):
        self.value = theta
        lnp = 0.
        for chain in chains:
            lnp_chain = ((1-self.params['pout']) * self.params['cluster'].integrate_chain(chain) +
                         self.params['pout'] * self.params['background'].integrate_chain(chain))
            lnp += lnp_chain

        return lnp_chain + self.lnp_prior


if __name__ == "__main__":

    parlist = []
    for par in ['age', 'dist']:
        mname, sname = 'mu_{}'.format(par), 'sigma_{}'.format(par)
        sample = Parameter(par, prior=GaussianPrior([mname, sname]), free=False)
        mu = Parameter(mname, prior=UniformPrior())
        sigma = Parameter(sname, prior=LogarithmicPrior())

        parlist += [sample, mu, sigma]

    mass = Parameter('mass', prior=PowerLawPrior(['gamma_imf']))
    gamma = Parameter('gamma_imf', free=False)
    gamma.value = -2.35

    parlist += [mass, gamma])
    model = Likelihood(parlist)
