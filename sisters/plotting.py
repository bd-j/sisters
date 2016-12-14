import numpy as np
import matplotlib.pyplot as pl

def plot_stars(stars, param='age', thin=5):
    zero = np.zeros(len(stars[0][param])) + 0.5
    fig, axes = pl.subplots(2, 2)

    for i, ax in enumerate(axes.flat):
        ax.hist(stars[i][param], bins=30,
                histtype='stepfilled', alpha=0.5)
        for x in stars[i][param][::thin]:
            ax.plot([x,x], [10,20], color='k')

    return fig, axes


def plot_chain(chain):
    """
    """
    fig, axes = pl.subplots(1, 2)
    nw, ni, nd = chain.shape
    for j, ax in enumerate(axes.flat):
        for i in range(nw):
            ax.plot(chain[i, :, j])

    return fig, axes


def plot_star_chain(star_chains):
    fig, ax = pl.subplots(2, 2)
    for i, s in enumerate(star_chains):
        ax.flat[i % 4].hist(s['age'], histtype='step',
                            alpha =0.5, label='{}'.format(i))

    [a.legend(loc=0, prop={'size': 8}) for a in ax.flat]
    return fig, ax
