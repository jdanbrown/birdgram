# Setup python env
```sh
$ conda env create -f environment.yml
$ source activate bubo-features
$ pip install -U -r requirements.txt # Run manually so we can separate the conda/pip steps in docker build
$ pip install -e .
# $ Rscript --vanilla Rdepends.R # FIXME This takes forever; disabled here and in Dockerfile
```

## If osx: Manually install additional deps
- For ubuntu these are handled in Dockerfile
```sh
$ brew install ffmpeg --with-libvorbis --with-sdl2 --with-theora
```

## Optional: Update python env for local potoo/joblib dev
```sh
$ pip install -e .../potoo
$ pip install -e .../joblib
```

## Optional: Update python env after changing `environment.yml` or `requirements.txt`
```sh
$ conda env update -f environment.yml
$ pip install -U -r requirements.txt # Run manually to workaround https://github.com/pypa/pip/issues/2837
```

# Run

```sh
# Run dev api locally
$ bin/api-run-dev

# Run prod api remotely (example)
$ bin/gcloud-run --reuse=bubo-0 --disk-mode=rw --preemptible --machine-type=n1-standard-4 --container-pull --container-push 'time bin/api-cache-warm && bin/api-run-prod'
```

# HOWTO

## Hunt down recent data/cache/ (or data/xc/) files written recently
- Oops, I just wrote many gigs of intermediate cache data on my laptop that I meant to keep on the remote
- (And only sync the final api/mobile payloads to my laptop)
- (osx only)
```sh
mdfind -onlyin data/ 'kMDItemContentType != public.folder && kMDItemContentModificationDate >= $time.now(-86400)' \
  | sort \
  | sed -E ' s#(.*/(audio/xc/data|api/recs/recs_cache_audio_slices|joblib/datasets/com_name_to_species_dict|joblib/load/_metadata|joblib/load/recs|payloads/search_recs)[^/]*/).*$#\1#; ' \
  | uniq -c
```
