data_dir = '/Users/danb/hack/bubo/data'
cache_dir = f'{data_dir}/cache'

standard_sample_rate_hz = 22050  # Can resolve 11025Hz (by Nyquist), which most/all birds are below
default_log_ylim_min_hz = 512  # Most/all birds are above 512Hz (but make sure to clip noise below 512Hz)

unk_species = '_UNK'
unk_species_com_name = 'Unknown'
unk_species_taxon_id = 'TC___UNK'
unk_species_species_code = '___unk'

no_species = '_NON'
no_species_com_name = 'No species'
no_species_taxon_id = 'TC___NON'
no_species_species_code = '___non'
