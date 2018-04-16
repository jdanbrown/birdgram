#!/usr/bin/env python

# TODO
#   - Manually scrape https://academy.allaboutbirds.org/peterson-field-guide-to-bird-sounds/... -> data/...
#       - Already have full spectros
#   - Manually scrape http://www.xeno-canto.org/species/Zonotrichia-atricapilla?view=3&pg=1 -> data/xeno-canto/...
#       - Only have spectros with first 10s, have to compute full spectros ourselves
#   - Btw, here's the ebird taxon api for autocomplete: https://ebird.org/ws1.1/ref/taxon/find/CL27562?locale=en&q=sparrow

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

wav_sets.append({
    'id': 'xeno-canto',
    'uris': [
      {
        # TODO How to incorporate xeno-canto?
        #   - Already very rich, but needs focus, efficiency, purpose, better workflow, etc.
        #       - http://www.xeno-canto.org/explore?query=%22Greater+Roadrunner+%22+q%3AA&view=3
        #   - Easy api:
        #       - http://www.xeno-canto.org/api/2/recordings?query=%22Greater+Roadrunner+%22+q%3AA&view=3
        #       - Docs: http://www.xeno-canto.org/article/153
        #
        # Can't stream directly because of CORS, would need to proxy (useful anyway e.g. for caching)
        # 'wavUri': 'http://www.xeno-canto.org/376218/download',
        'wavUri': 'data/xeno-canto/376218/audio.mp3',
        # How to get /uploaded/<foo>/ffts/XS<id>-full.png from http://www.xeno-canto.org/<id> ?
        #   - Would have to trim junk off the image anyway -- simpler to generate the specs ourselves on the fly?
        #   - Can get from html <meta> in http://www.xeno-canto.org/376218/
        'imgUri': 'http://www.xeno-canto.org/sounds/uploaded/MGVGHKBMIZ/ffts/XC376218-full.png',
        'wavMTime': '2016-07-09T11:38:51',
      },
      {
        'wavUri': 'data/xeno-canto/65830/audio.mp3',
        'imgUri': 'http://www.xeno-canto.org/sounds/uploaded/XEIROMUDEB/ffts/XC65830-full.png',
        'wavMTime': '2016-07-09T11:38:51',
      },
    ],
})

out_path = re.sub('\.py$', '', __file__)
with open(out_path, 'w') as f:
    json.dump(wav_sets, f, indent=2)
print('-> %s' % out_path)
