from functools import wraps
import itertools
from typing import *

import matplotlib.pyplot as plt
from more_itertools import first
import numpy as np
from potoo.pandas import df_transform_index
from potoo.plot import *
from potoo.util import round_sig
import scipy

from cache import cache
import metadata
from util import *

# TODO Very chewy, lots of room to improve:
#   - [x] Add @cache to img_foo()'s
#   - [x] Add progress=True (via tqdm) to img_foos() / plot_foos(), without lots of duplicated code
#   - [ ] Simplify: make a reusable pattern for the common args (progress, raw/scale, audio, ...)
#   - [ ] Simplify: refactor plot_foo(show=False) -> img_foo() called by plot_foo(), so caller can img_foo() directly


#
# Utils
#


def _with_many(plot_f: Callable[['X'], 'Y']) -> Callable[['X'], 'Y']:
    """
    Given plot_f(rec), add plot_f.many(recs)
    - Also adds another layer of caching, with granularity the whole input (i.e. recs)
    """

    @cache(
        version=4,
        tags='recs',
        key=lambda recs, *args, **kwargs: (recs.id, args, kwargs),
        nocache=lambda *args, show=True, **kwargs: show,  # No cache iff side effects iff show
    )
    def plot_many(
        recs,
        *args,
        show=True,
        raw=True,
        progress=dict(use='dask', scheduler='threads'),  # Faster than sync and processes (measured on plot_thumbs)
        **kwargs,
    ) -> Iterable['Y']:
        xs = list(df_rows(recs))
        f = lambda rec: first([
            plot_f(rec, *args, show=show, raw=raw, **kwargs),
            plt.show() if not raw else None,
        ])
        if show:
            # Side effects, avoid map_progress because (1) order is nondet and (2) dask bag runs f 1+n times (for meta)
            return list(map(f, xs))
        else:
            # No side effects, safe to par with map_progress
            return map_progress(f, xs,
                desc=plot_f.__qualname__,
                n=len(xs),
                **progress,
            )

    plot_f.many = plot_many
    plot_f.many.__name__ = plot_f.__name__ + '.many'
    plot_f.many.__qualname__ = plot_f.__qualname__ + '.many'

    return plot_f


#
# Spectros
#


@_with_many
def plot_thumb(
    rec,
    features,
    raw=True,
    scale=None,
    pad=True,  # Pad if duration_s < thumb_s (for consistent img size)
    audio=True,
    show=True,
    plot_kwargs=dict(),
    thumb_s=1,
    **thumb_kwargs,  # [Simpler to make (few, stable) thumb_kwargs explicit and let (many, evolving) **plot_kwargs be implicit?]
) -> Union[None, PIL.Image.Image, 'Displayable']:
    rec = rec_thumb(rec, features, thumb_s=thumb_s, **thumb_kwargs)
    return plot_spectro(rec, raw=raw, scale=scale, audio=audio, show=show, **plot_kwargs,
        pad_s=thumb_s if pad else None,
    )


@_with_many
def plot_slice(
    rec,
    features,
    slice_s=None,
    **kwargs,
) -> Union[None, PIL.Image.Image, 'Displayable']:
    if slice_s:  # Avoid excessive .slice() ops, else cache misses [Simpler to make slice_spectro noop/idempotent?]
        rec = features.slice_spectro(rec, 0, slice_s)
    return plot_spectro(rec, **kwargs)


def plot_spectro(
    rec,
    pad_s=None,
    audio=True,
    audio_kwargs=None,
    **kwargs,
) -> Union[None, PIL.Image.Image, 'Displayable']:
    ret = plot_spectros(rec, **kwargs,
        limit_s=pad_s and max(pad_s, rec.duration_s),  # Pad to pad_s, but don't limit <duration_s (e.g. 10.09s bug)
    )
    if audio and ret:
        # NOTE config.audio.audio_to_url.audio_kwargs will lossily compress audio without changing the spectro (above)
        #   - If you want to visualize lossy audio encoding via spectros, don't rely on this
        ret = display_with_audio(ret, audio=rec.audio.unbox, **(audio_kwargs or {}))
    return ret


@_with_many
def plot_spectro_wrap(
    rec,
    features,  # TODO Reimpl so we don't need features: slice up spectro.S instead of re-spectro'ing audio slices
    wrap_s=10,  # Sane default (5 too small, 20 too big, 10/15 both ok)
    limit_s=None,  # Limit[/pad?] spectro duration
    pad=True,  # Pad to wrap_s if there's only one line (for consistent img size)
    audio=True,
    audio_kwargs=None,
    **kwargs,
) -> Union[None, PIL.Image.Image, 'Displayable']:
    limit_s = min(limit_s or np.inf, rec.duration_s)
    wrap_s = wrap_s or limit_s
    n_wraps = int(np.ceil(limit_s / wrap_s))
    breaks_s = np.linspace(0, n_wraps * wrap_s, n_wraps, endpoint=False)
    ret = plot_spectros(
        pd.DataFrame(
            [rec] if list(breaks_s) == [0] else  # Avoid excessive .slice() ops in your ids, else cache misses (HACK)
            [features.slice_spectro(rec, b, min(b + wrap_s, limit_s)) for b in breaks_s]
        ),
        yticks=['%.0fs' % s for s in breaks_s],
        xformat='+%.0fs',
        limit_s=wrap_s if pad else None,
        **kwargs,
    )
    if audio and ret:
        # NOTE config.audio.audio_to_url.audio_kwargs will lossily compress audio without changing the spectro (above)
        #   - If you want to visualize lossy audio encoding via spectros, don't rely on this
        ret = display_with_audio(ret, audio=rec.audio.unbox, **(audio_kwargs or {}))
    return ret


@cache(
    version=0,
    tags='recs',
    key=lambda recs, *args, **kwargs: (recs.id, args, kwargs),
    nocache=lambda *args, show=True, **kwargs: show,  # No cache iff side effects iff show
)
def plot_spectros(
    recs: Union[pd.DataFrame, Row],
    raw=True,
    title=None,
    xformat='%.0fs',
    ytick_col=None,  # e.g. 'species_longhand'
    yticks=None,
    limit_s=None,  # Limit/pad spectro duration
    verbose=False,
    **kwargs,
) -> Optional[PIL.Image.Image]:
    """Vis lots of spectros per dataset"""

    # Allow recs or rec
    if isinstance(recs, pd.DataFrame):
        recs = list(df_rows(recs))
    else:
        recs = [recs]

    Ss = [rec.spectro.S for rec in recs]
    ts = [rec.spectro.t for rec in recs]
    limit_s = limit_s or max([rec.duration_s for rec in recs])  # Allow limit_s > duration_s, to allow padding (for consistent img size)
    dt = ts[0][1] - ts[0][0]  # Not exactly right, but close enough
    limit_i = int(round(limit_s / dt))
    f_i_len = Ss[0].shape[0]
    Ss = [S[:, :limit_i] for S in Ss]
    ts = [t[:limit_i] for t in ts]
    pad_first = np.array([[0, 0], [0, 0]])  # [top, bottom], [left, right]
    pad_rest  = np.array([[0, 1], [0, 0]])  # [top, bottom], [left, right]
    # Flip vertically (twice) to align with plt.imshow(..., origin='lower')
    cat_Ss = np.flip(axis=0, m=np.concatenate([
        np.flip(axis=0, m=np.pad(S, mode='constant', constant_values=np.nan, pad_width=(
            (pad_first if i == 0 else pad_rest) +
            np.array([[0, 0], [0, limit_i - S.shape[1]]])
        )))
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
        return show_img(cat_Ss, origin='lower', file_prefix=title or 'image', **raw_kwargs)
    else:
        # (imshow is faster than pcolormesh by ~4x)
        plt.imshow(cat_Ss, origin='lower')
        plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda t_i, pos=None: xformat % (t_i * dt)))
        if title:
            plt.title(title, loc='left')
        if not ytick_col and yticks is None:
            plt.yticks([])
        else:
            if yticks is None:
                yticks = [rec[ytick_col] for rec in recs]
            plt.yticks(
                np.arange(len(recs)) * (f_i_len + pad_rest[0].sum()) + f_i_len // 2,
                # Reverse to align with plt.imshow(..., origin='lower')
                reversed(yticks),
            )


#
# Spectro grids
#   - These have fallen out of favor since we added df_cell and can now viz spectros directly within df tables
#   - Keeping them around for reference, for now
#


def plot_thumb_grid(
    recs,
    features,
    raw=True,
    cols=None,
    order='top-down',  # Thumbs are typically longer than tall, so default grid order to top-down instead of left-right
    plot_kwargs=dict(),
    **thumb_kwargs,  # [Simpler to make (few, stable) thumb_kwargs explicit and let (many, evolving) **plot_kwargs be implicit?]
):
    recs = recs_thumb(recs, features, **thumb_kwargs)
    return plot_spectro_grid(recs, raw=raw, cols=cols, order=order, **plot_kwargs)


def plot_spectro_grid(
    recs,
    raw=True,
    cols=None,
    order='top-down',  # Spectros are typically longer than tall, so default grid order to top-down instead of left-right
    **kwargs,
):
    # HACK-y reuse of plot_patches
    #   - TODO Simplify/refactor/whatever
    n = len(recs)
    if cols is None:
        cols = int(np.sqrt(n))
    Ss = [s.S for s in recs.spectro]
    f = max(S.shape[0] for S in Ss)
    t = max(S.shape[1] for S in Ss)
    # Pad Ss to a uniform shape
    Ss = np.array([
        np.pad(S, mode='constant', constant_values=np.nan, pad_width=np.array([
            [0, f - S.shape[0]],  # [top, bottom]
            [0, t - S.shape[1]],  # [left, right]
        ]))
        for S in Ss
    ])
    return plot_patches(
        patches=np.array([
            S.reshape(t * f)  # HACK Because plot_patches expects its input to be flattened
            for S in Ss
        ]).T,
        f_bins=f,
        patch_length=t,
        raw=raw,
        rows=int(np.ceil(n / cols)),
        order=order,
        **kwargs,
    )


def plot_patches(
    patches,
    f_bins,
    patch_length,
    raw=False,
    rows=None,
    sort={},
    order='left-right',  # 'left-right' | 'top-down'
    **kwargs,
):
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
    patches = patches.reshape(rows, cols, f, p,
        order={
            'left-right': 'C',
            'top-down': 'F',
        }[order],
    )
    assert patches.shape == (rows, cols, f, p)

    # Pad each patch with a 1px bottom/right border: (f,p) -> (f+1, p+1)
    patches = np.array([
        np.array([
            np.pad(patch, mode='constant', constant_values=np.nan, pad_width=np.array([
                [0, 1],  # [top, bottom]
                [0, 1],  # [left, right]
            ]))
            for patch in row
        ])
        for row in patches
    ])
    assert patches.shape == (rows, cols, f+1, p+1)

    # Flip row axis for origin='lower' (below) so that patches start in upper left (instead of lower left)
    patches = np.flip(patches, axis=0)

    # Paste everything into a 2D image
    image = np.concatenate(axis=1, seq=np.concatenate(axis=1, seq=patches))

    # Pad with a top/left border
    image = np.pad(image, mode='constant', constant_values=np.nan, pad_width=np.array([
        [1, 0],  # [top, bottom]
        [1, 0],  # [left, right]
    ]))

    assert image.shape == (1 + rows*(f+1), 1 + cols*(p+1))

    # Plot
    kwargs.setdefault('origin', 'lower')  # Run y-axis upward 0->f, not downward f->0
    if raw:
        show_img(image, **kwargs)
    else:
        plt.imshow(image, **kwargs)


#
# Confusion matrix
#


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


#
# PCA
#


def plot_pca_var_pct(pca: 'sk.decomposition.PCA', filter_df=None) -> ggplot:
    return (
        pd.DataFrame(dict(var_pct=pca.explained_variance_ratio_))
        .reset_index()
        .assign(
            component=lambda df: df.index,
            cum_var_pct=lambda df: df.var_pct.cumsum(),
        )
        .pipe(lambda df: df[filter_df] if filter_df is not None else df)
        .pipe(ggplot)
        + aes(x='component')
        + geom_point(aes(y='var_pct'), fill='none')
        + geom_line(aes(y='var_pct'), alpha=.5)
        + geom_point(aes(y='cum_var_pct'), color='blue', fill='none')
        + geom_line(aes(y='cum_var_pct'), color='blue', alpha=.5)
        + scale_y_continuous(limits=(0, 1), breaks=np.linspace(0, 1, 10+1))
        + coord_flip()
        + theme_figsize(aspect=1/4)  # (Caller can override simply by adding another theme_figsize to the result)
    )


# TODO Oops, did we ~duplicate plot_pca_var_pct? How do these differ? Should we keep both?
def plot_pca_var_explained(
    var_explained: np.ndarray,  # Cumulative variance explained, per component
    max_n_components: float = np.inf,
    max_var_explained: float = 1,
) -> ggplot:
    """Plot PCA var_explained ~ n_components"""
    return (
        pd.DataFrame([
            dict(n_components=i + 1, var_explained=var_explained)
            for i, var_explained in enumerate(var_explained)
        ])
        [lambda df: df.n_components <= max_n_components]
        [lambda df: df.var_explained <= max_var_explained]
        .pipe(ggplot)
        + aes(x='n_components', y='var_explained')
        + geom_point()
        + theme_figsize(aspect=1/6)
        + scale_x_continuous(minor_breaks=[], breaks=lambda lims: np.arange(round(lims[0]), round(lims[1]) + 1))
        + ggtitle(f'PCA var_explained ({len(var_explained)} components)')
    )


def plot_all_projections(
    df: pd.DataFrame,
    ndims: int,  # How many dimures to generate pairs for, e.g. ndims=4 -> [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    dim_name: Union[str, Callable[[int], str]],  # Dimension column name in df, as a function of dimension index
    pack=False,  # Pack cells instead of leaving ~half blank
    diag_hack=False,  # Plot densities on the diagonal [TODO Make this less hacky and brittle]
    nrows: int = None,
    ncols: int = None,
    sharex=True,
    sharey=True,
    diag_kwargs=dict(),
    width: float = None,
    aspect: float = None,
    title_kwargs=dict(),
    suptitle=None,
    suptitle_top=.94,  # Manually adjust top margin for tight_layout + suptitle (ugh)
    lim=(-1, 1),  # HACK TODO FIXME How to make sns/plt compute this automatically from the data? (What did I obstruct?)
    xlim=None,
    ylim=None,
    **scatter_kwargs,
):
    """
    Similar in spirit to sns.pairplot, except:
    - Omit plots below the diagonal, for ~2x speedup
    - Allow sharex/sharey
    - Allow pack=True, for 2x visual density

    Example usage:

        plot_all_projections(
            df_pca,
            ndims=10,
            dim_name='x%s',
            hue='species',
            palette='tab10',
            pack=True,
        )

    Similar results from sns.pairplot, for reference:

        sns.pairplot(
            data=df_pca[['species', *[f'x{i}' for i in range(10)]]],
            hue='species',
            diag_kind='kde',
            markers='.',
            plot_kws=dict(
                linewidth=0,
                s=10,
            ),
            diag_kws=dict(
                shade=True,
                linewidth=1,
                alpha=.5,
            ),
        )

    """

    # Lazy import
    import seaborn as sns

    # Defaults
    scatter_kwargs.setdefault('marker', '.')  # Tiny markers
    scatter_kwargs.setdefault('s', 10) # plt 's' instead of sns 'size', else sns includes the single size value in the legend
    scatter_kwargs.setdefault('linewidth', 0)  # Don't draw edges around markers
    diag_kwargs = dict(diag_kwargs)  # Don't mutate
    diag_kwargs.setdefault('linewidth', 1)
    title_kwargs = dict(title_kwargs)  # Don't mutate
    title_kwargs.setdefault('fontsize', 8)
    if xlim is None: xlim = lim
    if ylim is None: ylim = lim

    # Interpret params
    if isinstance(dim_name, str):
        dim_name = lambda i, dim_name=dim_name: dim_name % i
    dim_pairs = [(a, b - 1) for a, b in combinations(range(ndims + 1), 2)]
    if not pack:
        nrows = nrows or ndims
        ncols = ncols or ndims
    elif not nrows and not ncols:
        ncols = int(np.ceil(np.sqrt(len(dim_pairs))))
    if not nrows:
        nrows = len(dim_pairs) // ncols + 1
    elif not ncols:
        ncols = len(dim_pairs) // nrows + 1
    ncells = nrows * ncols
    figsize = theme_figsize(width=width, aspect=aspect or nrows / ncols).figsize

    # Pre-compute densities
    #   - HACK HACK HACK Taking lots of shortcuts here to solve my current use case and move on. Very brittle. Keep
    #     hacky bits gated behind diag_hack=True, off by default.
    if diag_hack:
        densities = []
        for ci in range(ndims):
            c = dim_name(ci)
            hue = scatter_kwargs.get('hue', None)
            groups = df.groupby(hue) if hue else [(None, df)]
            densities.append([
                [xs, ys]
                for name, g in groups
                for xs in [np.linspace(df[c].min() * 1.2, df[c].max() * 1.2, 100)]
                for ys in [scipy.stats.kde.gaussian_kde(g[c])(xs)]
            ])
        # Transform ys to fit within ylim
        y_max = max(
            np.max(ys)
            for groups in densities
            for (xs, ys) in groups
        )
        for groups in densities:
            for g in groups:
                g[1] = g[1] * (ylim[1] - ylim[0]) / (y_max * 1.05) + ylim[0]

    # Plot
    xmin = xmax = ymin = ymax = 0
    fig, axes = plt.subplots(nrows, ncols, sharex=sharex, sharey=sharey, figsize=figsize)
    if (nrows, ncols) == (1, 1):
        axes = {(0, 0): axes}
    for i in range(ncells):
        (r, c) = (i // ncols, i % ncols)
        ax = axes[r, c]
        legend = 'full' if (r, c) == (nrows - 1, 0) else False  # Show legend only once, in the (empty) bottom-left cell
        if pack:
            j = i
        else:
            row_starts = np.cumsum([0, *np.arange(ndims - 1, 0, -1)])
            j = row_starts[r] + c if r < ndims and c < ndims else None
        if j is not None and j < len(dim_pairs) and (pack or c >= r):
            (yi, xi) = dim_pairs[j]
            (y, x) = (dim_name(yi), dim_name(xi))
            if x != y:
                sns.scatterplot(ax=ax, data=df, x=x, y=y, **scatter_kwargs, legend=legend)
                ax.axhline(color='black', alpha=.1, zorder=-1)
                ax.axvline(color='black', alpha=.1, zorder=-1)
                ax.set_title('%s,%s' % (yi, xi), **title_kwargs)
                xmin = min(xmin, df[x].min())
                xmax = max(xmax, df[x].max())
                ymin = min(ymin, df[y].min())
                ymax = max(ymax, df[y].max())
            elif diag_hack:
                for xs, ys in densities[xi]:
                    ax.plot(xs, ys, **diag_kwargs)
                ax.set_title('%s' % xi, **title_kwargs)
                ax.axvline(color='black', alpha=.1, zorder=-1)
        elif legend:
            # If we're on the last cell but out of dim_pairs, draw a blank plot to ensure the legend gets drawn
            (yi, xi) = dim_pairs[0]
            (y, x) = (dim_name(yi), dim_name(xi))
            sns.scatterplot(data=df, x=x, y=y, ax=ax, **{**scatter_kwargs, 'marker': ''}, legend=legend)
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Limits
    plt.xlim(xlim)
    plt.ylim(ylim)

    # Aesthetics
    sns.despine()  # Just top,right by default (pass more params for left,bottom)
    plt.tight_layout()

    if suptitle:
        fig.suptitle(suptitle)
        fig.subplots_adjust(top=suptitle_top)  # Manually adjust top margin for tight_layout + suptitle (ugh)
