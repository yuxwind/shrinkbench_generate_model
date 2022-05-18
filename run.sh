#!/bin/bash
path=`pwd`
export PYTHONPATH="$path:$PYTHONPATH"
dataset=$1
arch=$2
round=$3
gpu=$4
log=logs/$dataset-$arch-round$round.log
echo $log
python generate_models.py  $dataset $arch $round $gpu > $log 2>&1 &
#tail -f $log