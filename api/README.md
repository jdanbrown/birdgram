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
$ docker build . -t jdanbrown/bubo-api:latest
```

Run prod from docker image:
- TODO Why barfs without `-it`?
```sh
$ docker run -p8000:8000 -it bubo-api bin/run-prod
```

Push docker image to dockerhub:
```sh
$ docker push jdanbrown/bubo-api:latest
```
