#!/usr/bin/env bash

./build.sh

docker save tigerexamplealgorithm | gzip -c > tigerexamplealgorithm.tar.xz
