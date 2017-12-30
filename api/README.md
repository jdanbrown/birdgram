Run dev:
```sh
bin/run-dev
```

See dev data:
```sh
python api/ebird.py
http get localhost:8000/nearby_barcharts | jq .[] -c
```

Run prod:
```sh
bin/run-prod
```

Login to dockerhub:
```sh
docker login
```

Build docker base layers:
```sh
VERSION=... # e.g. v0
docker build . --file Dockerfile-base -t bubo/api-base:"$VERSION"
docker push bubo/api-base:"$VERSION"
```

Build docker image:
```sh
docker build . -t bubo/api
```

Run prod from docker image:
```sh
docker run -p8000:8000 -it bubo/api bin/run-prod
```

Push docker image to dockerhub:
```sh
docker push bubo/api
```

Auth `kubectl` with GKE:
- [Warning: bad creds get cached and must be manually removed](https://github.com/kubernetes/kubernetes/issues/38075)
```sh
gcloud auth application-default login
# Maybe also edit ~/.kube/config as per https://github.com/kubernetes/kubernetes/issues/38075
kubectl --context=... version
```

Deploy:
```sh
KUBECTL_CONTEXT=... bin/deploy
```

Query:
```sh
http 104.197.235.14/health
http 104.197.235.14/nearby_barcharts
http 104.197.235.14/focus-birds/v0
```

See logs:
```sh
kubectl --context=... logs -f api-<tab>
```
