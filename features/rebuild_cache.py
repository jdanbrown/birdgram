##
# TODO Clean up local cache for full-duration .spectro -- ate up ~130/160g of my free disk space
#   - Plot distribution of `ll data/cache/joblib/sp14/model/_spectro_cache\(version\=0\)/00*/output.pkl`, kill the bigger mode

## {time: 3.726s}
from notebooks import *
# memory.log.level = 'debug'  # XXX Debug
sg.init(app=None)

## {time: 1.946s}
recs = (sg.xc_meta
    # .sample(n=3, random_state=0)  # XXX Debug
    .pipe(recs_featurize_audio_meta)
    .pipe(df_inspect, lambda df: len(df))
)

##
# Cache full recs (wav .audio + .spectro + .feat)
#   - [x] local
#   - [x] remote
# for ix in tqdm(list(chunked(recs.index, 1000))):
#     out = (recs
#         .loc[ix]
#         # .pipe(df_inspect, lambda df: df.id[:3])  # Debug
#         # Load .audio
#         .pipe(sg.load.audio, use='dask', scheduler='threads')
#         # Load .spectro (for sliced .audio)
#         .assign(spectro=lambda df: sg.features.spectro(df,
#             scheduler='threads',
#             cache=True,  # Do cache .spectro calls (unlike model training where we only need them for caching .feat)
#         ))
#         # Load .feat (for sliced .spectro)
#         .pipe(sg.projection.transform)  # .feat <- .spectro
#         # Persist .audio (compressed for app, not uncompressed for model fit)
#         #   - After .spectro/.feat, to mimic app behavior (it serves .spectro/.feat from audio id before .enc(mp4), not after)
#         #   - XXX Only for sliced, not for full
#         # .pipe(recs_audio_ensure_has_file)
#         # .pipe(df_inspect, lambda df: df.id[:3])  # Debug
#     )

##
# Cache 10s slices (mp4 .audio + .spectro + .feat)
#   - [x] local
#   - [TODO] remote
#       - QA:
#           $ find data/cache/audio/xc/data/ -type f | pv -terbl >/tmp/cache-audio-files
#           $ cat /tmp/cache-audio-files | wc -l
#           $ cat /tmp/cache-audio-files | ag '.enc\(wav\)$' | wc -l
#           $ cat /tmp/cache-audio-files | ag '.enc\(mp4,libfdk_aac,32k\)$' | wc -l
#       - [Overnight: stopped after 18000/35227...]
for ix in tqdm(list(chunked(recs.index, 1000))):
    print()  # Preserve tqdm state before progress bars (below) overwrite it
    out = (recs
        .loc[ix]
        # .pipe(df_inspect, lambda df: df.id[:3])  # Debug
        # Load .audio
        #   - TODO load.audio is the current bottleneck on 100% cache hit (e.g. when resuming a partially failed run)
        #       - Don't need to load .audio when there's no work to do anywhere downstream
        #       - Fixing this would require rotating this from horizontal (bulk) to vertical (one row at a time, e2e)...
        .pipe(sg.load.audio, use='dask', scheduler='threads')
        # Slice .audio
        #   - .slice_audio i/o .slice_spectro to mark (.spectro_denoise() in audio id) that .spectro is computed after the slice i/o before
        .pipe(df_map_rows_progress, desc='slice_audio',
            use='dask', scheduler='threads',
            f=lambda row: sg.features.slice_audio(row, 0, 10,
                recompute_spectro=False,  # We skipped loading the original .spectro, and we'll compute the sliced .spectro ourselves
            ),
        )
        # .pipe(df_inspect, lambda df: df.id[:3])  # Debug
        # Load .spectro (for sliced .audio)
        .assign(spectro=lambda df: sg.features.spectro(df,
            scheduler='threads',
            cache=True,  # Do cache .spectro calls (unlike model training where we only need them for caching .feat)
        ))
        # Load .feat (for sliced .spectro)
        .pipe(sg.projection.transform)  # .feat <- .spectro
        # Persist .audio (compressed for app, not uncompressed for model fit)
        #   - After .spectro/.feat, to mimic app behavior (it serves .spectro/.feat from audio id before .enc(mp4), not after)
        .pipe(recs_audio_ensure_has_file,
            load_audio=False,  # Skip lots of audio (ffmpeg) reads we don't need, e.g. on cache hit
        )
        # .pipe(df_inspect, lambda df: df.id[:3])  # Debug
    )

##
# # Test
# (out
#     .pipe(df_assign_first, **dict(
#         micro=df_cell_spectros(plot_spectro_micro.many, sg.features,
#             wrap_s=10,  # (= audio_s)
#             # audio=False,  # Don't include <audio> (from display_with_audio)
#             **{  # (= plot_many_kwargs)
#                 'scale': dict(h=int(40 * 1)),  # (scale=1)
#                 'progress': dict(use='dask', scheduler='threads'),
#                 '_nocache': True,  # Dev: disable plot_many cache since it's blind to most of our sub-many code changes
#             }
#         ),
#     ))
#     [:3]
# )

##
