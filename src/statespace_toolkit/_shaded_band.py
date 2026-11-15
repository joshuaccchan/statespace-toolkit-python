import numpy as np


def shaded_band(x, lo, hi, shade=0.85, ax=None):
    """ssm.shaded_band - a pointwise credible band drawn as a shaded region.

      h = ssm.shaded_band(x, lo, hi)
      h = ssm.shaded_band(x, lo, hi, shade)
      h = ssm.shaded_band(x, lo, hi, shade, ax)

      x      : points on the horizontal axis, numeric or dates
      lo, hi : lower and upper bounds at those points, the same length as x
      shade  : gray level in [0, 1] (default .85)
      ax     : matplotlib axes to draw on (default the current axes)
      h      : the PolyCollection drawn

    Plot lines after the band to draw them over it. Code-identical to
    bvar_toolkit.util.shaded_band of the Python bvar-toolkit, and the two must stay so; the same
    computation as shaded_band.m in the book's code for Chapter 14.

    See:
    Chan, J.C.C. (2027). Bayesian Macroeconometrics: Methods and Applications, Chapman &
    Hall/CRC.
    """
    import matplotlib.pyplot as plt

    ax = plt.gca() if ax is None else ax
    return ax.fill_between(np.ravel(x), np.ravel(lo), np.ravel(hi), facecolor=(shade,) * 3,
                           edgecolor='none', linewidth=0)
