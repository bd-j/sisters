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


def subtriangle(chain, parnames, outname=None, showpars=None,
                start=0, thin=1, truths=None, trim_outliers=None,
                extents=None, **kwargs):
    """Make a triangle plot of the (thinned, latter) samples of the posterior
    parameter space.  Optionally make the plot only for a supplied subset of
    the parameters.

    :param start:
        The iteration number to start with when drawing samples to plot.

    :param thin:
        The thinning of each chain to perform when drawing samples to plot.

    :param showpars:
        List of string names of parameters to include in the corner plot.

    :param truths:
        List of truth values for the chosen parameters
    """
    try:
        import triangle
    except(ImportError):
        import corner as triangle

    nw, ni, nd = chain.shape
    s0 = int(ni * start)
    # pull out the parameter names and flatten the thinned chains
    flatchain = chain[:, start::thin, :]
    flatchain = flatchain.reshape(flatchain.shape[0] * flatchain.shape[1],
                                  flatchain.shape[2])

    # restrict to parameters you want to show
    if showpars is not None:
        ind_show = np.array([p in showpars for p in parnames], dtype=bool)
        flatchain = flatchain[:, ind_show]
        #truths = truths[ind_show]
        parnames = parnames[ind_show]
    if trim_outliers is not None:
        trim_outliers = len(parnames) * [trim_outliers]
    try:
        fig = triangle.corner(flatchain, labels=parnames, truths=truths,  verbose=False,
                              quantiles=[0.16, 0.5, 0.84], extents=trim_outliers, **kwargs)
    except:
        fig = triangle.corner(flatchain, labels=parnames, truths=truths,  verbose=False,
                              quantiles=[0.16, 0.5, 0.84], range=trim_outliers, **kwargs)

    if outname is not None:
        fig.savefig('{0}.triangle.png'.format(outname))
        #pl.close(fig)
    else:
        return fig
