Run dev:
```sh
bin/run-dev
```

Run prod:
```sh
bin/run-prod
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
gcloud auth login
gcloud auth application-default login
kubectl version
```

Deploy:
```sh
bin/deploy
```

Query:
```sh
http 104.197.235.14/health
http 104.197.235.14/barchart
```

See logs:
```sh
kubectl logs -f api-<tab>
```
