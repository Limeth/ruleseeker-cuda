#!/usr/bin/env bash
./include-shaders.sh
nvcc *.cu -lGLEW -lglut -lGL -lpng -o a.out -O3 -g -G -DPNG_NO_SETJMP $@
