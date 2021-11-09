#!/usr/bin/env bash
./include-shaders.sh
nvcc *.cu --gpu-architecture=sm_35 -lGLEW -lglut -lGL -lpng -o build-debug -O3 -g -G -DPNG_NO_SETJMP $@
