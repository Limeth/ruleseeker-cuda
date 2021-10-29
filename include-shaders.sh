#!/usr/bin/env bash
# This script converts the shaders
HEADER_FILE="shaders.cuh"
rm -f $HEADER_FILE

echo "#pragma once" >>$HEADER_FILE

for SHADER_FILE in `ls config.h *.vert *.frag`; do
    xxd -i $SHADER_FILE >>$HEADER_FILE
done
