import calendar
import itertools
from typing import Callable, Iterable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from potoo.pandas import df_reverse_cat, df_transform_index
from potoo.plot import *
from potoo.util import round_sig, tap

import metadata
from util import *


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
        show_img(cat_Ss, origin='lower', file_prefix=query_and_title or 'image')
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


def plot_barchart(
    barcharts_df,
    downsample_sp=None,
    cols=None,
    width_per_col=4,
    aspect_per_col=1,
    random_state=0,
    debug=False,
) -> ggplot:

    # Defaults (which you'll usually want to tweak)
    downsample_sp = downsample_sp or df.species.nunique()
    cols = cols or min(3, downsample_sp // 100 + 2)

    cmap = (
        # Debug: multi-color palette
        mpl.colors.ListedColormap(np.array(mpl.cm.tab10.colors)[[3, 1, 6, 4, 5, 7, 0, 9, 2]]) if debug else
        # Match color from ebird barcharts, for visual simplicity
        mpl.colors.ListedColormap(9 * ['#31c232'])
    )

    return (barcharts_df

        # Downsample by species (optional)
        .pipe(lambda df: df[df.species.isin(df.species
            .drop_duplicates().sample(replace=False, random_state=random_state, n=downsample_sp)
        )])

        # Add longhand, com_name (via join)
        .set_index('species', drop=False)
        .join(how='left', other=metadata.species.df.set_index('shorthand')[['longhand', 'com_name']])

        # Add w (ebird bar width) from p (prob)
        .assign(w=lambda df: df.p.map(ebird_bar_width))

        # Add facet_col
        .assign(facet_col=lambda df: (
            cols * df.species.cat.remove_unused_categories().cat.codes.astype(int) // df.species.nunique()
        ))

        # Debug: inspects
        .pipe(df_inspect, lambda df: () if not debug else (
            df_summary(df).T,
            df[:3],
            # Plot dist(p, w)
            (ggplot(df)
                + aes(x='p', fill='factor(w)') + geom_histogram(breaks=[*reversed(list(ebird_bar_widths)), 1])
                + theme_figsize(aspect=1/6) + xlim(0, 1) + scale_fill_cmap_d(cmap)
            ),
        ))

        # Plot
        .pipe(df_reverse_cat, 'species', 'longhand', 'com_name')
        .pipe(ggplot)
        + facet_wrap('facet_col', nrow=1, scales='free_y')
        + aes(x='week', y='longhand', color='factor(w)', fill='factor(w)')
        + geom_point(aes(size='w'), shape='|', stroke=2)
        + scale_size_radius(
            range=(-1.5, 4),  # Min/max bar height for geom_point(size) (Default: (1, 6))
        )
        + scale_fill_cmap_d(cmap)
        + scale_color_cmap_d(cmap)
        + scale_x_continuous(limits=(1, 48),
            expand=(0, 1),  # Minimal expand (mul, add)
            labels=[5 * ' ' + x[0] for x in calendar.month_abbr[1:]] + [''],  # 12+1 month labels, on the major breaks
            breaks=np.arange(1, 48+4, 4) - .5,  # 13 grey major breaks, so we can label them with months
            minor_breaks=np.arange(1, 48+4, 4*3) - .5,  # 5 black minor breaks, to display over the major breaks
        )
        + theme(
            legend_position='none',
            axis_title=element_blank(),
            axis_text_x=element_text(size=7),
            axis_text_y=element_text(size=7),
            axis_ticks_major=element_blank(),
            panel_background=element_blank(),
            panel_grid_major_x=element_line(color='lightgrey', size=.5),  # Major/minor are inverted (see above)
            panel_grid_minor_x=element_line(color='black', size=.5),
            panel_grid_major_y=element_blank(),
            panel_grid_minor_y=element_blank(),
            strip_text=element_blank(),
            panel_spacing_x=2,
        )
        + theme_figsize(width=width_per_col * cols, aspect=aspect_per_col * cols)

    )


ebird_bar_widths = {
    # https://help.ebird.org/customer/portal/articles/1010553-understanding-the-ebird-bar-charts
    # https://help.ebird.org/customer/en/portal/articles/1210247-what-is-frequency-
    .600: 9,
    .400: 8,
    .300: 7,
    .200: 6,
    .100: 5,
    .050: 4,
    .010: 3,
    .003: 2,
    .000: 1,
}


def ebird_bar_width(p: float, bar_widths=ebird_bar_widths) -> int:
    return next(v for k, v in bar_widths.items() if p >= k)
