#!/usr/bin/env bash
./include-shaders.sh
nvcc src/*.cu -lGLEW -lglut -lGL -lpng -lbsd -o build-release -O3 -DNDEBUG -DPNG_NO_SETJMP $@
