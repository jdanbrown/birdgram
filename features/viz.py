import matplotlib.pyplot as plt
import numpy as np
from potoo.plot import *

from sp14.model import Model


def plot_many_spectros(
    recs,
    query_and_title=None,
    raw=False,
    spectro_col='spectro',
    ytick_col='species_longhand',
    t_max=60,
    n=None,
    verbose=True,
):
    """Vis lots of spectros per dataset"""

    if query_and_title:
        recs = recs.query(query_and_title)

    (Ss, ts) = (recs
        .pipe(lambda df: df.query(query_and_title) if query_and_title else df)
        [:n]
        .pipe(lambda df: (
            df.spectro.map(lambda s: s.S),
            df.spectro.map(lambda s: s.t),
        ))
    )
    dt = ts.iloc[0][1] - ts.iloc[0][0]  # This isn't exactly right, but it's close enough
    t_i_max = int(np.ceil(t_max / dt))
    f_i_len = Ss.iloc[0].shape[0]
    Ss = Ss.map(lambda S: S[:, :t_i_max])
    ts = ts.map(lambda t: t[:t_i_max])
    pad = np.array([[0, 1], [0, 0]])  # [top, bottom], [left, right]
    cat_Ss = (Ss
        .map(lambda S: np.pad(
            S,
            pad + np.array([[0, 0], [0, t_i_max - S.shape[1]]]),
            'constant',
            constant_values=np.nan,
        ))
        # Flip vertically (twice) to align with plt.imshow(..., origin='lower')
        .pipe(lambda s: np.flip(axis=0, m=np.concatenate([np.flip(axis=0, m=S) for S in s], axis=0)))
    )
    if verbose:
        print(f'cat_Ss.shape[{cat_Ss.shape}]')

    if raw:
        # Output image resolution is 1-1 with input array shape, but we can't add axis labels and figure titles
        plot_img_raw(cat_Ss, origin='lower', file_prefix=query_and_title or 'image')
    else:
        # (imshow is faster than pcolormesh by ~4x)
        plt.imshow(cat_Ss, origin='lower')
        plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda t_i, pos=None: '%.0fs' % (t_i * dt)))
        if query_and_title:
            plt.title(query_and_title, loc='left')
        if not ytick_col:
            plt.yticks([])
        else:
            plt.yticks(
                np.arange(len(recs)) * (f_i_len + pad[0].sum()) + f_i_len // 2,
                # Reverse to align with plt.imshow(..., origin='lower')
                reversed(list(recs[ytick_col])),
            )


def plot_centroids(model: Model):
    """Viz a model's projection patch centroids (pca -> skm)"""

    # Unpack input
    skm = model.proj_skm_
    p = model.config.patch_config.patch_length

    # Compute total projection: pca -> skm.D
    proj = (skm.D.T @ skm.pca.components_).T

    # Dimensions
    (fp, k) = proj.shape
    f = fp // p

    width = get_figsize()['width']
    height = get_figsize()['height']
    rows = np.sqrt(k / ((f+1)/height / ((p+1)/width)))
    rows = int(np.floor(rows))  # Arbitrary choice: floor (taller) vs. ceil (wider)
    cols = int(np.ceil(k / rows))
    # display(((rows, cols), rows * cols))

    centroids = np.array([
        np.array([
            proj[i*f:(i+1)*f, j]
            for i in range(p)  # Inner loop: p patch strides
        ]).T
        for j in range(k)  # Outer loop: k centroids
    ])
    # display(centroids.shape)

    centroids_extend = np.append(
        centroids,
        np.full((rows * cols - k, f, p), np.nan),
        axis=0,
    )
    # display(centroids_extend.shape)

    centroids_pad = np.array([np.pad(
        x,
        np.array([[0, 1], [0, 1]]),  # [top, bottom], [left, right]
        'constant',
        constant_values=np.nan,
    ) for x in centroids_extend])
    # display(centroids_pad.shape)

    centroids_table = centroids_pad.reshape(rows, cols, f+1, p+1)
    # display(centroids_table.shape)

    centroids_layout = np.pad(
        np.concatenate(np.concatenate(centroids_table, axis=1), axis=1),
        np.array([[1, 0], [1, 0]]),  # [top, bottom], [left, right]
        'constant',
        constant_values=np.nan,
    )
    plt_show_img(centroids_layout)
