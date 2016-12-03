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
$ docker build . -t bubo-api
```

Run prod from docker image:
- TODO Why barfs without `-it`?
```sh
$ docker run -p8000:8000 -it bubo-api bin/run-prod
```
