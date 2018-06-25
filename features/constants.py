import os

# TODO Find a cleaner way to do this
_maybe_code_dirs = [
    '/Users/danb/hack/bubo/features',  # Local dev (osx)
    '/app',  # Remote deploy (docker/linux)
]
for _maybe_code_dir in _maybe_code_dirs:
    if os.path.exists(_maybe_code_dir):
        code_dir = _maybe_code_dir
        break
else:
    raise Exception(f'Found none of these dirs to use as code_dir: {_maybe_code_dirs}')

data_dir = f'{code_dir}/data'
cache_dir = f'{data_dir}/cache'

standard_sample_rate_hz = 22050  # Can resolve 11025Hz (by Nyquist), which most/all birds are below
default_log_ylim_min_hz = 512  # Most/all birds are above 512Hz (but make sure to clip noise below 512Hz)

unk_species = '_UNK'
unk_species_com_name = 'Unknown'
unk_species_taxon_id = 'TC___UNK'
unk_species_species_code = '___unk'

mul_species = '_MUL'
mul_species_com_name = 'Multiple species'
mul_species_taxon_id = 'TC___MUL'
mul_species_species_code = '___mul'

no_species = '_NON'
no_species_com_name = 'No species'
no_species_taxon_id = 'TC___NON'
no_species_species_code = '___non'
