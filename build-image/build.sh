#!/bin/bash

set -e  # exit on error

VERSION=3
REGISTRY=ic-registry.epfl.ch
IMG_NAME=swiss-ai/goldfish
FULL_IMG_NAME=$REGISTRY/$IMG_NAME

# Setup the builder

# Check if builder exists
if ! docker buildx ls | grep -q multiplatform-builder; then
    echo "Creating new builder: multiplatform-builder"
    docker buildx create --name multiplatform-builder \
        --driver-opt env.BUILDKIT_STEP_MAX_SIZE=2147483648 \
        --driver-opt env.BUILDKIT_MAX_PARALLEL_JOBS=1 \
        --driver-opt env.BUILDKIT_STEP_LOG_MAX_SIZE=10485760 \
        --use
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
    --build-arg DOCKER_BUILDKIT=1 \
    --build-arg PIP_NO_CACHE_DIR=1 \
    --build-arg MAX_JOBS=1 \
    -f "./DockerfileRCP" \
    .