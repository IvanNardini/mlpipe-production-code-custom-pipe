#!/bin/bash

cd ./oop

for script in $*; do
    if [ $script == 'test' ]; then
        python3 test.py
    fi
done