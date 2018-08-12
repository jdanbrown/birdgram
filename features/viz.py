import itertools
from typing import Callable, Iterable, Optional, Union

import matplotlib.pyplot as plt
from more_itertools import first
import numpy as np
from potoo.pandas import df_transform_index
from potoo.plot import *
from potoo.util import round_sig

import metadata
from util import *


def plot_spectro_micro(
    x: Union['rec', 'spectro'],
    features,
    wrap=True,  # False to truncate at wrap_s
    wrap_s=15,
    limit_s=None,
    **kwargs,
):
    # TODO Pad (like plot_spectros) if audio < wrap_s, for uniform img height across various recs
    kwargs.setdefault('raw', True)
    kwargs.setdefault('show_audio', True)
    if not wrap and not limit_s:
        limit_s = wrap_s
    if limit_s:
        x = features.with_audio(x, lambda audio: audio[:limit_s * 1000])
    plot_spectro_wrap(x, features, wrap_s=wrap_s, **kwargs)


def plot_spectro_wrap(
    x: Union['rec', 'spectro'],
    features,  # TODO Reimpl so we don't need features: slice up spectro.S instead of re-spectro'ing audio slices
    wrap_s: float = 10,  # Sane default (5 too small, 20 too big, 10/15 both ok)
    pad=True,  # In case there's only one line, pad out to wrap_s, e.g for uniform height when raw=True
    show_audio=False,
    raw=False,
    **kwargs,
):
    spectro = x.spectro if hasattr(x, 'spectro') else x
    wrap_ms = int(wrap_s * 1000)
    len_audio_ms = len(spectro.audio)
    n_wraps = int(np.ceil(len_audio_ms / wrap_ms))
    breaks_ms = np.linspace(0, n_wraps * wrap_ms, n_wraps, endpoint=False)
    if pad:
        kwargs['t_max'] = wrap_s
    plot_spectros(
        pd.DataFrame(
            dict(spectro=features.with_audio(spectro, lambda audio: audio[i : i + wrap_ms]))
            for i in breaks_ms
        ),
        yticks=['%.0fs' % (ms / 1000) for ms in breaks_ms],
        xformat=lambda s: '+%.0fs' % s,
        raw=raw,
        **kwargs,
    )
    if show_audio:
        if not raw:
            plt.show()
        display(spectro.audio)


def plot_spectro(
    x: Union['rec', 'spectro'],
    raw=False,
    show_audio=False,
    **kwargs,
):
    rec = x if hasattr(x, 'spectro') else pd.Series(dict(spectro=x))
    plot_spectros(pd.DataFrame([rec]), raw=raw, **kwargs)
    if show_audio:
        if not raw:
            plt.show()
        display(rec.spectro.audio)


def plot_spectros(
    x: Union['recs', 'spectros'],
    query_and_title=None,
    raw=False,
    spectro_col='spectro',
    xformat=lambda x: '%.0fs' % x,
    ytick_col=None,  # e.g. 'species_longhand'
    yticks=None,
    t_max='auto',  # Pass an int (seconds) to manually set max time (x-axis)
    n=None,
    verbose=False,
    **kwargs,
):
    """Vis lots of spectros per dataset"""

    recs = x if hasattr(x, 'spectro') else pd.DataFrame(dict(spectro=x))
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
    if t_max == 'auto':
        t_max = recs.spectro.map(lambda spectro: len(spectro.audio)).max() / 1000
    t_max_i = first(list(np.where(ts.iloc[0] > 3)[0]) or [len(ts.iloc[0])])
    f_i_len = Ss.iloc[0].shape[0]
    Ss = Ss.map(lambda S: S[:, :t_max_i])
    ts = ts.map(lambda t: t[:t_max_i])
    pad_first = np.array([[0, 0], [0, 0]])  # [top, bottom], [left, right]
    pad_rest  = np.array([[0, 1], [0, 0]])  # [top, bottom], [left, right]
    # Flip vertically (twice) to align with plt.imshow(..., origin='lower')
    cat_Ss = np.flip(axis=0, m=np.concatenate([
        np.flip(axis=0, m=np.pad(
            S,
            (pad_first if i == 0 else pad_rest) + np.array([[0, 0], [0, t_max_i - S.shape[1]]]),
            'constant',
            constant_values=np.nan,
        ))
        for i, S in enumerate(Ss)
    ]))
    if verbose:
        print(f'cat_Ss.shape[{cat_Ss.shape}]')

    if raw:
        # Output image resolution is 1-1 with input array shape, but we can't add axis labels and figure titles
        raw_kwargs = {
            **(raw if isinstance(raw, dict) else {}),  # New api (simpler to compose)
            **kwargs,  # Back compat
        }
        show_img(cat_Ss, origin='lower', file_prefix=query_and_title or 'image', **raw_kwargs)
    else:
        # (imshow is faster than pcolormesh by ~4x)
        plt.imshow(cat_Ss, origin='lower')
        dt = ts.iloc[0][1] - ts.iloc[0][0]  # Not exactly right, but close enough
        plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda t_i, pos=None: xformat(t_i * dt)))
        if query_and_title:
            plt.title(query_and_title, loc='left')
        if not ytick_col and yticks is None:
            plt.yticks([])
        else:
            if yticks is None:
                yticks = list(recs[ytick_col])
            plt.yticks(
                np.arange(len(recs)) * (f_i_len + pad[0].sum()) + f_i_len // 2,
                # Reverse to align with plt.imshow(..., origin='lower')
                reversed(yticks),
            )


def plot_patches(patches, f_bins, patch_length, raw=False, rows=None, sort={}, **kwargs):
    """Viz a set of patches (f*p, n) as a grid that matches figsize"""

    # Dimensions
    f = f_bins
    p = patch_length
    (fp, n) = patches.shape
    assert fp == f * p, f'fp[{fp}] != f[{f}] * p[{p}], for patches.shape[{patches.shape}]'

    # Compute (rows, cols) for patch layout, based on figsize (w,h) and patch size (f,p)
    if not rows:
        width = get_figsize()['width']
        height = get_figsize()['height']
        rows = np.sqrt(n / ((f+1)/height / ((p+1)/width)))
        rows = max(1, int(np.floor(rows)))  # Arbitrary choice: floor (taller) vs. ceil (wider)
    cols = int(np.ceil(n / rows))

    # Unstack each patch: (f*p, n) -> (n,f,p)
    patches = np.moveaxis(patches.reshape(f, p, n), 2, 0)

    # Sort (optional)
    if sort:
        patches = np.array(sorted(patches, **sort))

    # Extend patches to complete the grid: n -> rows*cols
    patches = np.append(
        patches,
        np.full((rows * cols - n, f, p), np.nan),
        axis=0,
    )
    assert patches.shape == (rows * cols, f, p)

    # Wrap into a grid: (rows*cols, ...) -> (rows, cols, ...)
    patches = patches.reshape(rows, cols, f, p)
    assert patches.shape == (rows, cols, f, p)

    # Pad each patch with a 1px bottom/right border: (f,p) -> (f+1, p+1)
    patches = np.array([
        np.array([
            np.pad(
                patch,
                np.array([[0, 1], [0, 1]]),  # [top, bottom], [left, right]
                'constant',
                constant_values=np.nan,  # nan renders as white
            ) for patch in row
        ])
        for row in patches
    ])
    assert patches.shape == (rows, cols, f+1, p+1)

    # Paste everything into a 2D image
    image = np.pad(
        np.concatenate(
            np.concatenate(
                # Flip row axis for origin='lower' (below) so that patches read left->right, top->bottom
                np.flip(patches, axis=0),
                axis=1,
            ),
            axis=1,
        ),
        # Pad with a top/left border
        np.array([[1, 0], [1, 0]]),  # [top, bottom], [left, right]
        'constant',
        constant_values=np.nan,
    )
    assert image.shape == (1 + rows*(f+1), 1 + cols*(p+1))

    # Plot
    kwargs.setdefault('origin', 'lower')  # Run y-axis upward 0->f, not downward f->0
    if raw:
        show_img(image, **kwargs)
    else:
        plt.imshow(image, **kwargs)


def sort_species_confusion_matrix(
    df: pd.DataFrame,
    dtype=metadata.species.df.shorthand.dtype,
) -> pd.DataFrame:
    return (df
        .pipe(df_transform_index, lambda c: c.astype(dtype)).sort_index()
        .T.pipe(df_transform_index, lambda c: c.astype(dtype)).sort_index().T
    )


def plot_confusion_matrix(M: np.ndarray, classes: Iterable[str], **kwargs):
    plot_confusion_matrix_df(
        pd.DataFrame(M, columns=classes, index=classes),
        **kwargs,
    )


def plot_confusion_matrix_df(
    df: pd.DataFrame,
    title: str = None,
    title_y: float = 1.08,  # Fussy
    format: Optional[Union[str, Callable]] = lambda x: (('%s' % round_sig(x, 2)) if x < 1 else '%.1f' % x).lstrip('0') or '0',
    ylabel='y_true',
    xlabel='y_pred',
    marginals=True,
    normalize=True,
    raw=False,
    sort_df = sort_species_confusion_matrix,  # TODO Not our concern? Sure is convenient...
    **kwargs,
):
    """From http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html"""

    # Coerce format
    #   - TODO Skip the `for i, j` below if no format [have to default xticks/yticks to a sane default format]
    format = format or (lambda x: '')
    if isinstance(format, str):
        format = lambda x: str % x

    # Sort indexes and columns of matrix
    if sort_df:
        df = sort_df(df)

    # Data
    M = df.as_matrix()
    classes = list(df.index)
    if normalize:
        M = M.astype('float') / M.sum(axis=1)[:, np.newaxis]
        M = np.nan_to_num(M)

    # Add marginals to labels
    #   - Don't add them to grid, else their magnitude will drown out everything else
    xticks = classes
    yticks = classes
    if marginals:
        xticks = [' '.join([tick, '(%s)' % format(x)]) for tick, x in zip(xticks, M.sum(axis=0))]
        yticks = [' '.join(reversed([tick, '(%s)' % format(x)])) for tick, x in zip(yticks, M.sum(axis=1))]

    # Plot
    kwargs.setdefault('origin', 'upper')
    if raw:
        show_img(M, **kwargs)
    else:
        plt.imshow(M, interpolation='nearest', **kwargs)
        plt.gca().xaxis.tick_top()
        plt.gca().xaxis.set_label_position('top')
        plt.xticks(range(len(xticks)), xticks, rotation=90)
        plt.yticks(range(len(yticks)), yticks)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if title:
            plt.title(title, y=title_y)
        plt.tight_layout()
        thresh = M.max() / 2.
        for i, j in itertools.product(range(M.shape[0]), range(M.shape[1])):
            plt.text(
                j,
                i,
                format(M[i,j]),
                horizontalalignment='center',
                color='white' if M[i,j] > thresh else 'black',
                # color='lightgray' if M[i,j] > thresh else 'darkgray',
                # color='gray',
            )
