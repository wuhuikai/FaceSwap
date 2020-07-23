#!/usr/bin/env bash

# Sourcing the secrets needed for the unit test (for local running only)
FILE=./.secrets/environment.env
if test -f "$FILE"; then
    echo "$FILE exist - Running locally"
    source ${FILE}
    export $(cut -d= -f1 $FILE)
fi

# Build the container
docker build --build-arg FRISBEE_TOKEN="${FRISBEE_TOKEN}" --build-arg SWAP_TOKEN="${SWAP_TOKEN}" -t faceswap_docker_build .
docker tag faceswap_docker_build registry.heroku.com/gary-robot/web

heroku container:login
docker push registry.heroku.com/gary-robot/web

# Release it to Heroku
heroku container:release web --app gary-robot
