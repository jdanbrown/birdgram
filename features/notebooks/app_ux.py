from constants import *
from util import *


def load_app_recs(
    projection,
    datasets=['recordings'],
    n=None,
    spectro=True,
    cache_spectro=True,
    **kwargs,
) -> 'recs':
    """Load recs + featurize: .audio, .feat"""

    # Load
    recs = (projection.features.load.recs(datasets=['recordings'])
        [:n]
    )

    # Featurize: .audio, .feat
    recs = (recs
        # Add .audio
        .assign(audio=lambda df: projection.features.load.audio(df, scheduler='threads'))
        # Add .recorded_at, .audio_sha, .audio_id
        #   - TODO Move these upstream into Load._metadata
        #       - Careful: Adding .stat into Load._metadata might add a bottleneck
        #       - Careful: Adding sha1hex(audio._data) into Load._metadata might add a bottleneck
        #       - Careful: Replacing rec.id with .audio_id will bust caches
        .assign(
            recorded_at=lambda df: pd.to_datetime(df.path.map(lambda x: (Path(data_dir) / x).stat().st_mtime) * 1e9),
            audio_sha=lambda df: df.audio.map(lambda x: sha1hex(x.unbox._data)),
            audio_id=lambda df: df.apply(axis=1, func=lambda row: '-'.join([
                row.recorded_at.date().isoformat().replace('-', ''),
                row.audio_sha[:4],
            ])),
        )
        .set_index('audio_id')
        # Add .feat
        .pipe(projection.transform, override_scheduler='synchronous')
        # Ergonomics
        .pipe(df_reorder_cols, first=['recorded_at'])
        .sort_values('recorded_at', ascending=False)
    )

    # De-dupe by audio_id
    _dupe_audio_id = (recs
        .reset_index()  # audio_id
        [lambda df: df.audio_id.isin(
            df.assign(n=1).groupby('audio_id').n.sum()[lambda s: s > 1].index
        )]
        .set_index('audio_id')
    )
    _n_recs_with_dupes = len(recs)
    recs = recs.groupby('audio_id').first()
    if len(_dupe_audio_id):
        log.warn('Dropped %s recs with duplicate audio_id' % (_n_recs_with_dupes - len(recs)))
        display(_dupe_audio_id)

    # Heavyweight
    if spectro:
        recs = app_recs_add_spectro(recs, projection.features, cache=cache_spectro)

    return recs


def app_recs_add_spectro(recs, features, **kwargs) -> 'recs':
    """Featurize: .spectro (slow)"""
    # Cache control is knotty here: _spectro @cache is disabled to avoid disk blow up on xc, but we'd benefit from it for recordings
    #   - But the structure of the code makes it very tricky to enable @cache just for _spectro from one caller and not the other
    #   - And the app won't have the benefit of caching anyway, so maybe punt and ignore?
    return (recs
        .assign(spectro=lambda df: features.spectro(df, scheduler='threads', **kwargs))  # threads >> sync, procs
    )
