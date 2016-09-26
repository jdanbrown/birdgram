#!/usr/bin/env python

from datetime import datetime
import json
import os
import os.path
import re

default_wav_to_img = lambda x: re.sub('^data/', 'data/spec/', x) + '.png'

wav_sets = []
for (id, wav_dir, wav_to_img) in [
    ('recordings',             'data/recordings/',                  default_wav_to_img),
    ('mlsp-2013',              'data/mlsp-2013-wavs/',              default_wav_to_img),
    ('nips4b',                 'data/nips4b-wavs/',                 default_wav_to_img),
    ('warblr-2016-ff1010bird', 'data/warblr-2016-ff1010bird-wavs/', default_wav_to_img),
    ('warblr-2016-warblrb10k', 'data/warblr-2016-warblrb10k-wavs/', default_wav_to_img),
    ('birdclef-2015',          'data/birdclef-2015-wavs/',          default_wav_to_img),
]:
    print('%s: %s ...' % (id, wav_dir), end=' ', flush=True)
    paths = os.listdir(wav_dir)
    paths.sort()
    wav_paths = []
    for path in paths:
        path = os.path.join(wav_dir, path)
        if os.path.isfile(path) and path.endswith('.wav'):
            wav_paths.append({
                'wavUri':   path,
                'imgUri':   wav_to_img(path),
                'wavMTime': datetime.fromtimestamp(os.stat(path).st_mtime).isoformat(), # UTC
            })
    print('%s wavs' % len(wav_paths))
    wav_sets.append({
        'id':   id,
        'uris': wav_paths,
    })

out_path = re.sub('\.py$', '', __file__)
with open(out_path, 'w') as f:
    json.dump(wav_sets, f, indent=2)
print('-> %s' % out_path)
