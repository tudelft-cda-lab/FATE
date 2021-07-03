#!/bin/zsh
clang++ -shared -Wall -fPIC -o3 $1 -o afl_mutate.so -I /home/cas/AFLplusplus/include