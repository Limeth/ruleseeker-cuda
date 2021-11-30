#!/usr/bin/env bash
./include-shaders.sh
nvcc src/*.cu -lGLEW -lglut -lGL -lpng -lbsd -o build-debug -O3 -g -G -DPNG_NO_SETJMP $@
