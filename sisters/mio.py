import numpy as np


def read_minesweeper(filename, nout=None, **extras):
    print(filename)
    with open(filename, 'r') as f:
        # drop the comment hash and mags field
        header = f.readline().split()
    data = np.genfromtxt(filename, comments='#', skip_header=1,
                         dtype=np.dtype([(n.lower(), np.float) for n in header]))

    p = np.exp(data['logwt'] - data['logz'][-1])
    samples = np.random.choice(data, p=p, size=nout)
    return samples


def load_stars(files, nout=None, **extras):
    return [read_minesweeper(f, nout=nout) for f in files]
