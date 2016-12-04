Run dev:
```sh
$ bin/run-dev
```

Run prod:
```sh
$ bin/run-prod
```

Build docker image:
```sh
$ docker build . -t bubo/api
```

Run prod from docker image:
- TODO Why barfs without `-it`?
```sh
$ docker run -p8000:8000 -it bubo/api bin/run-prod
```

Push docker image to dockerhub:
```sh
$ docker push bubo/api
```

Deploy:
```sh
GIT_SHA="`git rev-parse --short HEAD`"
docker build . -t bubo/api:"$GIT_SHA"
docker push bubo/api:"$GIT_SHA"
# TODO ...
# TODO -> bin/deploy
```
