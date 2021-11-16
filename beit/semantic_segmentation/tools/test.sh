#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
OUT=$3
EVAL=$4
SHOW_DIR=$5
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/test.py $CONFIG $CHECKPOINT --out $OUT --eval $EVAL --show-dir $SHOW_DIR
