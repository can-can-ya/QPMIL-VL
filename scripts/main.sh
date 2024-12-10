#!/bin/bash

# path to the config file
CONFIG=configs/main.yaml

# get time
CURRENT_TIME=$(date '+%Y-%m-%d %H:%M:%S')

cd ..

for SEED in 1 2 3 4 5 6 7 8 9 10
do
    python3 main.py \
    -f ${CONFIG} \
    -s ${SEED} \
    -t "${CURRENT_TIME}"
done