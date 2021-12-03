#!/usr/bin/env bash
./include-shaders.sh
nvcc src/*.cu -I/usr/local/cuda/include -lGLEW -lglut -lGL -lpng -lbsd -o build-release -O3 -DNDEBUG -DPNG_NO_SETJMP $@
