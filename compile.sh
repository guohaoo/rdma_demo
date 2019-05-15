#!/bin/bash
set -x

if [ "$1" == "reader" ]; then
  gcc -g demo_reader.c -I/usr/local  -o demo_reader \
    -L/usr/local/cuda/lib64 \
    -lpthread -libverbs -lrdmacm -lcuda -lcudart
else
  gcc -g demo_server.c -I/usr/local -o demo_server \
    -L/usr/local/cuda/lib64 \
    -lpthread -libverbs -lrdmacm -lcuda -lcudart
fi
