
set -ex

docker build -t space2dataset .
docker run --rm -p 8000:8000 --env-file .env --name test space2dataset
