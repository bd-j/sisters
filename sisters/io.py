import numpy as np
import gzip

__all__ = ["read_minesweeper", "load_stars"]


def read_minesweeper(filename, nout=None, **extras):
    """Read chains from minesweeper output, making corrections for the evidence
    at each iteration to produce a true chain
    """
    print(filename)
    if filename[-3:] == '.gz':
        with gzip.open(filename,'rb') as f:
            for line in f:
                header = line.split()
                break
    else:
        with open(filename, 'r') as f:
            # drop the comment hash and mags field
            for line in f:
                header = line.split()
                break
    dt = np.dtype([(n.lower(), np.float) for n in header])
    data = np.genfromtxt(filename, comments='#', skip_header=1,
                         dtype=dt)

    # Evidence based correction
    p = np.exp(data['logwt'] - data['logz'][-1])
    samples = np.random.choice(data, p=p, size=nout)
    return samples


def load_stars(files, nout=None, **extras):
    return [read_minesweeper(f, nout=nout) for f in files]
