import matplotlib as mpl
from skm import SKM


def cart_to_polar(x, y):
    z = x + y * 1j
    return (np.abs(z), np.angle(z))


def polar_to_cart(r, theta):
    z = r * np.exp(1j * theta)
    return (z.real, z.imag)


# A more plt friendly version of skm.display.visualize_clusters
def skm_visualize_clusters(skm: SKM, X, cmap=mpl.cm.Set1):
    """Visualize the input data X and cluster assignments from SKM model skm"""
    assert skm.assignment is not None
    if skm.do_pca:
        X = skm.pca.transform(X.T).T  # Whiten data to match centroids
    plt.axhline(0, color='lightgray')
    plt.axvline(0, color='lightgray')
    color_i = mpl.cm.ScalarMappable(
        cmap=cmap,
        norm=mpl.colors.Normalize(vmin=0, vmax=skm.k - 1),
    ).to_rgba
    for n, sample in enumerate(X.T):
        plt.plot(sample[0], sample[1], '.', color=color_i(skm.assignment[n]))
    for n, centroid in enumerate(skm.D.T):
        plt.plot(centroid[0], centroid[1], 'o', color=color_i(n), markersize=8, markeredgewidth=2, markeredgecolor='k')
    # Set square lims (because polar)
    max_lim = max([*plt.gca().get_xlim(), *plt.gca().get_ylim()])
    plt.xlim(-abs(max_lim), abs(max_lim))
    plt.ylim(-abs(max_lim), abs(max_lim))
