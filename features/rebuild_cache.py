## {time: 3.726s}
from notebooks import *
# memory.log.level = 'debug'  # XXX Debug
sg.init(app=None)

## {time: 1.946s}
recs = (sg.xc_meta
    # .sample(n=3, random_state=0)  # XXX Debug
    .pipe(recs_featurize_audio_meta)
)

## Cache .audio
#   - Mem safety: load=False
(recs
    .pipe(sg.load.audio, use='dask', scheduler='threads',
        load=False,  # Mem safe
    )
)

## Cache .spectro
#   - Mem safety: load=False
(recs
    .assign(spectro=lambda df: sg.features.spectro(df,
        scheduler='threads',
        cache=True,  # Do cache .spectro calls (unlike model training where we only need them for caching .feat)
        load=False,  # Mem safe
    ))
)

## Cache .feat
#   - Mem safety: .feat isn't very large
(recs
    .pipe(sg.projection.transform)  # .feat <- .spectro
)

##
