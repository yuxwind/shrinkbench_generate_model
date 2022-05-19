#!/bin/bash

arch=$1
server=$2

mkdir -p $arch/saved_models
mkdir -p $arch/results


if [ "$server" = "longclaw" ]; then
    remote=/uusoc/exports/scratch/xiny/project/shrinkbench/Aidan
    cp $remote/shrinkbench/saved_models/*$arch*  $arch/saved_models/
    cp $remote/shrinkbench/results/$arch/*   $arch/results/
else  
    if [ "$server" = "thor" ]; then
        remote=thor:/home/thor/xin
    elif [ "$server" = "odin" ]; then
        remote=odin:/home/odin/xin
    elif [ "$server" = "cassini" ]; then
        remote=planck@cassini:/home/planck/xin
    elif [ "$server" = "curiosity" ]; then
        remote=zeus@curiosity:/home/zenus/xin/projects
    elif [ "$server" = "bluefish" ]; then
        remote=jcao@bluefish:/uusoc/exports/scratch/jcao/echo/shrinkbench
    elif [ "$server" = "redfish" ]; then
        remote=jcao@redfish:/uusoc/exports/scratch/jcao/echo/shrinkbench
    fi 
    
    echo $arch
    echo $remote
    scp -r $remote/shrinkbench/saved_models/*$arch*  $arch/saved_models/
    scp -r $remote/shrinkbench/results/$arch/*   $arch/results/
fi
