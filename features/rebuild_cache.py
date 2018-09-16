## {time: 3.726s}
from notebooks import *
# memory.log.level = 'debug'  # XXX Debug
sg.init(app=None)

## {time: 1.946s}
recs = (sg.xc_meta
    # .sample(n=3, random_state=0)
    # .sample(n=5, random_state=0)
    # .sample(n=10, random_state=0)
    # .sample(n=1000, random_state=0)
    .pipe(recs_featurize_audio_meta)
)

## {time: 48.89s}
audio_evals = (
    (recs

        # Load .audio
        .pipe(sg.load.audio, use='dask', scheduler='threads')

        # Add .feat, .spectro from .audio
        .pipe(sg.projection.transform)  # .feat
        .assign(spectro=lambda df: sg.features.spectro(df,
            scheduler='threads',
            cache=True,  # Do cache .spectro calls (unlike model training where we only need them for caching .feat)
        ))

    )
)

##
