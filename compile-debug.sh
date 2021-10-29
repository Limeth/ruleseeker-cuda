#!/usr/bin/env bash
./include-shaders.sh
nvcc *.cu -lGLEW -lglut -lGL -o a.out -O3 -g -G $@
