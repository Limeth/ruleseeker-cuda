#!/usr/bin/env bash
./include-shaders.sh
nvcc *.cu -rdc=true --gpu-architecture=sm_35 -lcudadevrt -lGLEW -lglut -lGL -lpng -o build-release -O3 -DNDEBUG -DPNG_NO_SETJMP $@
