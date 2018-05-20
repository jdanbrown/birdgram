import matplotlib.pyplot as plt
import numpy as np
from potoo.plot import *


def plot_many_spectros(
    recs,
    query_and_title=None,
    spectro_col='spectro',
    ytick_col='species',
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
    # TODO vpad with alternating 0/1 (e.g. dotted or dashed) so that clipping stands out (e.g. mlsp-2013 >8KHz is all 0)
    pad = np.array([[0, 1], [0, 0]])  # [top, bottom], [left, right]
    cat_Ss = (Ss
        .map(lambda S: np.pad(S, pad + np.array([[0, 0], [0, t_i_max - S.shape[1]]]), 'constant', constant_values=0))
        # Flip vertically (twice) to align with plt.imshow(..., origin='lower')
        .pipe(lambda s: np.flip(axis=0, m=np.concatenate([np.flip(axis=0, m=S) for S in s], axis=0)))
    )
    if verbose:
        print(f'cat_Ss.shape[{cat_Ss.shape}]')

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
