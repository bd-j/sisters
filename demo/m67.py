import sys, os, glob
import numpy as np

from kith.model import GaussianPriorND, lnpostfn
from kith.io import load_stars

from emcee import EnsembleSampler

# Initialize the model
model = GaussianPriorND(dimensions=['age', 'dist'])

# parameters to use for emcee
rp = {'nwalkers': 64,
      'niter': 128,
      'nout': 2000, # number of samples to draw from the minesweeper chains
      }

# List of ascii files containing the chains
files = glob.glob('../data/m67/*.dat')
star_chains = load_stars(files, **rp)

# arguments to the lnpostfn
postkwargs = {'samples': star_chains,
              'model': model}
# Instantiate sampler with lnpostfn
esampler = EnsembleSampler(rp['nwalkers'], model.ndim, lnpostfn,
                           kwargs=postkwargs)

# Initial center for the walkers
initial = [0.5, 0.2, 800, 100]
# give the walkers initial parameter positions with 10% dispersion
initial = [np.random.normal(loc=i, scale=0.1 * i, size=(rp['nwalkers']))
           for i in initial]
initial = np.array(initial).T

# Now iterate the sampler
for i, result in enumerate(esampler.sample(initial, iterations=rp['niter'],
                                           storechain=True)):

        if (i % 10) == 0:
            print(i)


# Write out some statistics from the last half of the chains, and plot the
# walker evolution
half = max(rp['niter'] / 2, 100)
import matplotlib.pyplot as pl
for (n, c) in zip(model.theta_names, esampler.chain.T):
    print('{}: mean={}, rms={}'.format(n, c[half:, :].mean(), c[half:, :].std()))
    fig, ax = pl.subplots()
    for i in range(rp['nwalkers']):
        ax.plot(c[:, i])
    ax.set_ylabel(n)
    ax.set_xlabel('MCMC iteration')
    ax.axvline(half, linestyle=':', color='k')
    
# Make a corner plot (removing some burnin)
outsize = rp['nwalkers'] * (rp['niter'] - half)
try:
    import corner
    fig = corner.corner(esampler.chain[:, half:, :].reshape(outsize, esampler.dim))
except:
    pass

pl.show()
