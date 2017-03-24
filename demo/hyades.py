import sys, os, glob
import numpy as np

from sisters.model import GaussianPriorND, lnpostfn
from sisters.io import load_stars

from emcee import EnsembleSampler

model = GaussianPriorND(dimensions=['age'])

rp = {'nwalkers': 128,
      'niter': 256,
      'nout': 2000, # number of samples to draw from the minesweeper chains
      }
#NSP = No spectral info, SP= with Spectral info
files = glob.glob('../data/hyades/*_NSP.dat') 
star_chains = load_stars(files, **rp)

postkwargs = {'samples': star_chains,
              'model': model}
esampler = EnsembleSampler(rp['nwalkers'], model.ndim, lnpostfn,
                           kwargs=postkwargs)

initial = [0.5, 0.2]
initial = [np.random.normal(loc=i, scale=0.1 * i, size=(rp['nwalkers']))
           for i in initial]
initial = np.array(initial).T

for i, result in enumerate(esampler.sample(initial, iterations=rp['niter'],
                                           storechain=True)):

        if (i % 10) == 0:
            print(i)
