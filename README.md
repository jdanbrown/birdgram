# bubo
bird song classifier, ftw

![01112015092413_snowy-owl-flying-across-a-field-in-falling-snow](https://cloud.githubusercontent.com/assets/2320606/16715170/daec6e18-468d-11e6-94a9-35669e342fcf.jpg)

# Caffe notes

Build docker image with caffe + jupyter:
```sh
$ docker build docker/ -t caffe-jupyter:latest
```

Run notebook in docker image:
- `sh -c` to avoid "restarting kernel" failures, due to some exec conflict between docker and ipython:
  - https://github.com/ipython/ipython/issues/7062
  - https://github.com/jupyter-attic/docker-notebook/pull/6
```sh
$ docker run -it -v ~/hack/bubo:/root/bubo -p 8888:8888 caffe-jupyter:latest sh -c 'jupyter notebook --ip 0.0.0.0 --debug --no-browser'
```

TODO Enable cpu parallelism in caffe:
- http://stackoverflow.com/questions/31395729/how-to-enable-multithreading-with-caffe
- https://github.com/BVLC/caffe/issues/1539
- https://hub.docker.com/r/kaixhin/caffe/~/dockerfile/
- https://hub.docker.com/r/tleyden5iwx/caffe-cpu-master/~/dockerfile/

TODO Try googlenet
