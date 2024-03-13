# MlOps


Serving
```shell
docker compose -f serving/docker-compose.yml up --build --force-recreate
```
Web App
```shell
docker compose -f webapp/docker-compose.yml up --build --force-recreate
```

Reporting
```shell
docker compose -f reporting/docker-compose.yml up --build --force-recreate
```