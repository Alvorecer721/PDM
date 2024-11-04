#!/bin/bash

set -e  # exit on error

VERSION=1
REGISTRY=ic-registry.epfl.ch
IMG_NAME=swiss-ai/goldfish
FULL_IMG_NAME=$REGISTRY/$IMG_NAME

# Setup the builder

# Check if builder exists
if ! docker buildx ls | grep -q multiplatform-builder; then
    echo "Creating new builder: multiplatform-builder"
    docker buildx create --name multiplatform-builder --use
else
    echo "Using existing builder: multiplatform-builder"
    docker buildx use multiplatform-builder
fi
docker buildx inspect --bootstrap

docker buildx build \
    --platform linux/amd64 \
    --tag $FULL_IMG_NAME:$VERSION \
    --push \
    --network=host \
    .