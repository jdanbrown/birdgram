# Setup python env
```sh
$ conda env create -f environment.yml
$ source activate bubo-features
$ pip install -e .
```

## Update python env after changing `environment.yml` or `requirements.txt`
```sh
$ conda env update -f environment.yml
$ pip install -U -r requirements.txt # Run manually to workaround https://github.com/pypa/pip/issues/2837
```
