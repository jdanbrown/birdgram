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
