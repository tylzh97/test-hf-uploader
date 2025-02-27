
set -ex

docker build -t space2dataset .
docker run --rm -p 7860:7860 --env-file .env --name test space2dataset
