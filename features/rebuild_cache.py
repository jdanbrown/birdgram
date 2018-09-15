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

        # .sample(random_state=0,
        #     # n=1,
        #     # n=3,
        #     n=100,
        # )

        # Load .audio
        .pipe(sg.load.audio, use='dask', scheduler='threads')

        # Add .feat, .spectro from transcoded .audio
        .pipe(sg.projection.transform)  # .feat
        .assign(spectro=lambda df: sg.features.spectro(df,
            scheduler='threads',
            cache=True,
        ))

    )
)

##
