#!/bin/bash


echo "$1.cpp"

hipcc -o $1 $1.cpp -I$ROCM_PATH/include -L$ROCM_PATH/lib  -l hipblas
