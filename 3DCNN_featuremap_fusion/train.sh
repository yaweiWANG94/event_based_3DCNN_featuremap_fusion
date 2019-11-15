#!/bin/bash
# Usage:
# ./train.sh DATASET [options args to {train,test}_net.py]
# DATASET is 20 or 02
#
# Example:
# ./train.sh 02

LOG="experiments/logs/faster_rcnn_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"



DATASETS=$1

save_path="./models/vgg16f/kaist_train-all${DATASETS}-fusion/logs/fusion.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
LOG=""./models/vgg16f/kaist_train-all${DATASETS}-fusion/logs/fusion.txt.`date +'%Y-%m-%d_%H-%M-%S'`""


case $DATASETS in
   20)
	echo "datasets use 20"
	exec &> >(tee -a "$LOG")
	echo Logging output to "$LOG"

	python trainval_net.py --dataset kaist_train-all20-fusion --modal fusion --net vgg16f --bs 4 --nw 8 --cuda --use_tfb \
	;;
   02)
	echo "datasets use 02"
	exec &> >(tee -a "$LOG")
	echo Logging output to "$LOG"

	python trainval_net.py --dataset kaist_train-all02-fusion --modal fusion --net vgg16f --bs 4 --nw 8 --cuda --use_tfb \
	;;
  *)
	echo "No datasets given"
	exit
	;;
esac