#!/bin/bash

srun -p dev \
    --job-name="Training" \
    --exclusive \
    ./train.sh
