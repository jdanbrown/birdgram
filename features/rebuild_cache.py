#!/usr/bin/env python

##
# [defer] Push slice down to ffmpeg moderate speedup (~3x speedup for 10s vs. 50s, which is xc avg)
#   - WARNING Based on a quick attempt in load/util, it's very nontrivial
#       - Add limit_s param to load.audio, et al.
#       - Pass limit_s down to util.audio_from_file_in_data_dir
#       - But then all the load._transcode_audio resample/write logic gets gnarly...
#           - Many occurrences of audio_from_file_in_data_dir to update (and reason about / test / risk breaking)
#           - Adding a new .ffmpeg() op after the file ext will break the "Do we need to change the encoding?" logic...
#           - And likely another couple concerns to wrangle with that I can't forsee yet...
#       - Feasible only if we're ok with audio ids that aren't shared with features.slice, e.g.
#           - '.mp3.ffmpeg(atrim=duration=10).resample(...).enc(wav)' vs '.mp3.resample(...).enc(wav).slice(0,10000)'
#           - And thumbnailing will always be bottlenecked by a full audio load (else e.g. thumbnail 10s within first 30s)
#   - Working pydub/ffmpeg example
#       # Need global -filter_complex instead of simpler input/output -t because pydub adds our args at end, in globals position
#       #   - https://github.com/jiaaro/pydub/blob/v0.23.0/pydub/audio_segment.py#L684-L692
#       #   - https://ffmpeg.org/ffmpeg.html
#       #   - https://ffmpeg.org/ffmpeg-filters.html
#       audio_from_file_in_data_dir('xc/data/WIFL/292295/audio.mp3', parameters=['-filter_complex', '[0:a]atrim=duration=10'])

## {time: 3.726s}
from notebooks import *
log_levels({'load': 'WARN'})  # Silence file reads/writes
# memory.log.level = 'debug'  # XXX Debug
sg.init(app=None)

## {time: 1.946s}
recs = (sg.xc_meta
    # .sample(n=3, random_state=0)  # XXX Debug
    [lambda df: df.species == 'WIFL']  # XXX Debug
    .pipe(recs_featurize_metadata)
    .pipe(df_inspect, lambda df: len(df))
)

##
# Cache 10s slices (mp4 .audio + .spectro + .feat)
#   - [x] local
#   - [x] remote
#   - How to QA
#       $ find data/cache/audio/xc/data/ -type f | pv -terbl >/tmp/cache-audio-files
#       $ cat /tmp/cache-audio-files | wc -l
#       $ cat /tmp/cache-audio-files | ag '.enc\(wav\)$' | wc -l
#       $ cat /tmp/cache-audio-files | ag '.enc\(mp4,libfdk_aac,32k\)$' | wc -l
for ix in tqdm(list(chunked(
    # recs.index, 1000,  # Killed on remote n1-highcpu-16 during 27/36
    recs.index, 250,  # Mem safe on remote n1-highcpu-16
))):
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
        #   - .slice_audio i/o .slice_spectro to mark (.spectro_denoise() in id) that .spectro is computed after the slice i/o before
        .pipe(df_map_rows_progress, desc='slice_audio',
            use='dask', scheduler='threads',
            f=lambda row: sg.features.slice_audio(row, 0, 10),
        )
        # .pipe(df_inspect, lambda df: df.id[:3])  # Debug
        # Persist .audio (compressed for app, not uncompressed for model fit)
        #   - Before .spectro/.feat, to mimic app behavior (it serves .spectro/.feat from .enc(mp4) audio id)
        .pipe(recs_audio_persist,
            # load_audio=False,  # Skip lots of audio (ffmpeg) reads we don't need, e.g. on cache hit
            # TODO Only uses ~15% of n1-highcpu-16 b/c bottlenecked by short ffmpeg fork+exec -- try progress_kwargs(npartitions=8*cores)
        )
        # .pipe(df_inspect, lambda df: df.id[:3])  # Debug
        # Cache + load .spectro (for sliced .audio)
        .assign(spectro=lambda df: sg.features.spectro(df,
            scheduler='threads',
            cache=True,  # Do cache .spectro calls (unlike model training where we only need them for caching .feat)
        ))
        # Cache + load .feat (for sliced .spectro)
        .pipe(sg.projection.transform)  # .feat <- .spectro
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
#         # .pipe(recs_audio_persist)
#         # .pipe(df_inspect, lambda df: df.id[:3])  # Debug
#     )

##
# # Test 10s slices
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
