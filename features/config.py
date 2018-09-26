import os

from attrdict import AttrDict

env  = os.environ.get('ENV') or os.environ.get('FLASK_ENV')
role = os.environ.get('BUBO_ROLE')

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
            # 'notebook': 'info',   # Normal (char) cache logging
            'notebook': 'warn',   # Silence cache logging
        }[role or 'notebook'],
    ),

    # Default progress_kwargs (-> util.progress_kwargs.default)
    progress_kwargs=dict(
        default=dict(use='dask', scheduler='threads'),  # Sane default
        # override=dict(use='sync'),  # Debug
    ),

    hosts=dict(
        prod='35.230.68.91',
        local='192.168.0.195:8000',
    ),

    server_globals=dict(
        sg_load=dict(

            # search
            experiment_id='comp-l1-l2-na-ca',
            cv_str='split_i=0,train=34875,test=331,classes=331',
            search_params_str='n_species=331,n_recs=1.0',
            classifier_str='cls=ovr-logreg_ovr,solver=liblinear,C=0.001,class_weight=balanced',
            random_state=0,
            fix_missing_skm_projection_id='peterson-v0-26bae1c',

            # xc_meta
            # countries_k=None, com_names_k=None,   num_recs=None,  # All xc.metadata
            countries_k='na', com_names_k='ca',   num_recs=None,  # NA/CA
            # countries_k='na', com_names_k='dan5', num_recs=None,  # XXX Faster dev
            # countries_k='na', com_names_k='dan5', num_recs=10,    # XXX Faster dev

        )
    ),

    api=dict(
        recs=dict(

            # Global audio_s for precomputed search_recs
            #   - TODO How to support multiple precomputed search_recs so user can choose e.g. 10s vs. 5s?
            audio_s=10,

            progress_kwargs = dict(
                use='dask', scheduler='threads',  # Faster (primarily useful for remote, for now)
                # use=None,  # XXX Dev (disable par and silence progress bars to more easily see reads/writes)
            ),

        ),
    ),

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
            #   - notebook: Files are way faster (~instant) and more lightweight (~0 mem) than inline data urls for #
            #     displaying many audios at once (>>10)
            #   - api: Data urls don't require serving the resource
            'api':      'data',
            'notebook': 'file',
        }[role or 'notebook'],
    ),

)
