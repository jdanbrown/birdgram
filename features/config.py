import os

from attrdict import AttrDict

env  = os.environ.get('ENV') or os.environ.get('FLASK_ENV')
role = os.environ.get('BUBO_ROLE')

override_progress_kwargs = dict(
    # use=None,    # XXX Debug: no par, no progress bars
    # use='log_time',  # XXX Debug: no par, log start/done instead of progress bars
    # use='sync',  # XXX Debug: no par, yes progress bars
    # use='dask', scheduler='threads',  # XXX Debug par
    # use='dask', scheduler='synchronous',  # XXX Debug par
)

# Global mutable config (handle with care)
config = AttrDict(

    role=role,

    logging=dict(
        datefmt=(
            '%H:%M:%S' if role == 'notebook' or env == 'development' else
            '%Y-%m-%dT%H:%M:%S'
        ),
    ),

    cache=dict(
        log_level={
            'api':      'debug',  # Verbose (line) cache logging
            # 'notebook': 'debug',  # Verbose (line) cache logging
            # 'notebook': 'info',   # Normal (char) cache logging
            'notebook': 'warn',   # Silence cache logging
        }[role or 'notebook'],
    ),

    # Default progress_kwargs (-> util.progress_kwargs.default)
    progress_kwargs=override_progress_kwargs or dict(
        default=dict(use='dask', scheduler='threads'),  # Sane default
        # override=dict(use='sync'),  # Debug
    ),

    # use='sync', but allow override via override_progress_kwargs
    sync_progress_kwargs=override_progress_kwargs or dict(
        use='sync',
    ),

    hosts=dict(
        prod='35.230.68.91',
        local='192.168.0.195:8000',
    ),

    server_globals=dict(
        sg_load=dict(
            search=dict(
                experiment_id='comp-l1-l2-na-ca',
                cv_str='split_i=0,train=34875,test=331,classes=331',
                search_params_str='n_species=331,n_recs=1.0',
                classifier_str='cls=ovr-logreg_ovr,solver=liblinear,C=0.001,class_weight=balanced',
                random_state=0,
                fix_missing_skm_projection_id='peterson-v0-26bae1c',
            ),
            xc_meta=dict(
                # countries_k=None, com_names_k=None,   num_recs=None,  # All xc.metadata
                countries_k='na', com_names_k='ca',   num_recs=None,  # NA/CA
                # countries_k='na', com_names_k='dan5', num_recs=None,  # XXX Faster dev
                # countries_k='na', com_names_k='dan5', num_recs=10,    # XXX Faster dev
            ),
        )
    ),

    api=dict(
        recs=dict(

            search_recs=dict(
                params=dict(
                    # Global params for precomputed search_recs
                    version=1,   # Manually bump to invalidate cache
                    # limit=1000,  # XXX Faster dev (declared here for cache invalidation)
                    audio_s=10,  # TODO How to support multiple precomputed search_recs so user can choose e.g. 10s vs. 5s?
                    scale=1,     # XXX Unused, but maintain cache hit [TODO TODO Kill when we're ok with cache rebuild]
                ),
                cache=dict(
                    # Manually specify what config values invalidate the search_recs cache (ugh...)
                    key=dict(
                        show=[
                            # Fields to surface in the cached filename, for easy human cache management
                            'config.api.recs.search_recs.params',
                            'config.server_globals.sg_load.xc_meta',
                        ],
                        opaque=[
                            # Fields that are too big/complex to reasonably stuff into a human-usable filename
                            'config.server_globals.sg_load.search',
                            'config.audio.audio_persist',  # Avoid .audio_to_url b/c it invalidates on role (notebook vs. api)
                        ],
                    ),
                ),
            ),

            progress_kwargs=override_progress_kwargs or dict(
                use='dask', scheduler='threads',  # Faster (primarily useful for remote, for now)
                # use='sync'  # XXX Dev
                # use=None,  # XXX Dev (disable par and silence progress bars to more easily see reads/writes)
            ),

        ),
    ),

    audio=dict(

        audio_persist=dict(
            # Tuned in notebooks/audio_codecs_data_volume
            audio_kwargs=dict(
                # format='mp3', bitrate='32k',                    # Clips hi freqs (just barely)
                # format='mp4', bitrate='32k', codec='aac',       # Worse than libfdk_aac(32k)
                format='mp4', bitrate='32k', codec='libfdk_aac',  # Very hard to distinguish from wav, and 10x smaller (!)
                # format='wav',                                   # Baseline quality, excessive size
            ),
        ),

        audio_to_url=dict(
            url_type={
                # Tradeoffs:
                #   - notebook: Files are way faster (~instant) and more lightweight (~0 mem) than inline data urls for
                #     displaying many audios at once (>>10)
                #   - api: Data urls don't require serving the resource
                'api':      'data',
                'notebook': 'file',
            }[role or 'notebook'],
        ),

    ),

    load=dict(

        # Performance (600 peterson recs):
        #   - Scheduler: [TODO Measure -- 'threads' is like the outcome, like n-1 of the rest]
        #   - Bottlenecks (no_dask): [TODO Measure]
        #   - TODO Revisiting with ~87k xc recs...
        metadata_progress_kwargs=override_progress_kwargs or dict(
            # [Local]
            use='dask', scheduler='threads',    # Optimal for 600 peterson recs on laptop
            # [Remote]
            # Perf comparison:                             machine     cpu   disk_r     disk_w  disk_io_r  disk_io_w
            # use='dask', scheduler='threads',    # n1-standard-16   5-20%   1-5m/s  10-120m/s      10-50     50-500
            # use='dask', scheduler='processes',  # n1-standard-16  10-50%  5-20m/s    ~100m/s     50-200    300-600
            # use='dask', scheduler='processes', get_kwargs=dict(num_workers=os.cpu_count() * 2),
        ),

        # Performance (measured with .audio on 600 peterson recs):
        #   - Scheduler: no_dask[.85s], synchronous[.93s], threads[.74s], processes[25s]
        #   - Bottlenecks (no_dask):
        #          ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        #             600    0.303    0.001    0.312    0.001 audio_segment.py:108(read_wav_audio)
        #             600    0.170    0.000    0.170    0.000 {method 'read' of '_io.BufferedReader' objects}
        #               1    0.060    0.060    0.845    0.845 <string>:1(<module>)
        #           61176    0.018    0.000    0.039    0.000 {built-in method builtins.isinstance}
        #             600    0.015    0.000    0.015    0.000 {built-in method io.open}
        audio_progress_kwargs=override_progress_kwargs or dict(
            # use='dask', scheduler='threads',  # Optimal for cache hits (disk read), but not cache misses (ffmpeg)
            # use='dask', scheduler='processes', get_kwargs=dict(num_workers=os.cpu_count() * 2),  # FIXME Too quiet...
            use='dask', scheduler='processes', get_kwargs=dict(num_workers=os.cpu_count() * 2), partition_size=10,
        ),

    ),

)
