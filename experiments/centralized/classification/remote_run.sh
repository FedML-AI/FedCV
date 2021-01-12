#!/usr/bin/env bash

CD_PATH=$1
HOST=$2
EXECUTE_CMD=$3

echo $CD_PATH
echo $HOST
echo $EXECUTE_CMD

cmd="cd $CD_PATH ; $EXECUTE_CMD"
echo $cmd
ssh $HOST $cmd
