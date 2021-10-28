#!/usr/bin/env bash
nvcc *.cu -lglut -lGL -o a.out -O3 -DNDEBUG $@
