#!/bin/bash

set -e  # exit on error

VERSION=1
REGISTRY=ic-registry.epfl.ch

IMG_NAME=swiss-ai/goldfish

docker build . -t $IMG_NAME:$VERSION --network=host
docker tag $IMG_NAME:$VERSION $REGISTRY/$IMG_NAME:$VERSION
docker push $REGISTRY/$IMG_NAME:$VERSION
