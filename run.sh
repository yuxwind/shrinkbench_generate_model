#!/bin/bash
dataset=$1
arch=$2
round=$3
gpu=$4
mkdir logs
log=logs/$dataset-$arch-round$round.log
echo $log
python generate_models.py  $dataset $arch $round $gpu > $log 2>&1 &
sleep 3
tail -f $log
