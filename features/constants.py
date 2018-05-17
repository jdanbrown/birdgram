from attrdict import AttrDict

data_dir = '/Users/danb/hack/bubo/data'
cache_dir = f'{data_dir}/cache'

standard_sample_rate_hz = 22050  # Can resolve 11025Hz (by Nyquist), which most/all birds are below
default_log_ylim_min_hz = 512  # Most/all birds are above 512Hz (but make sure to clip noise below 512Hz)

unk_species = 'XXXX'
