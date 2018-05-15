import attr


Recording = attr.make_class('Recording', [
    'dataset',
    'name',
    'species',
    'species_query',
    'basename',
    'duration_s',
    'samples_mb',
    'samples_n',
    'samples',
    'audio',
])
