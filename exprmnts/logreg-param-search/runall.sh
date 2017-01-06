#! /usr/bin/env bash

CONFIG_FILE=logreg.conf
END_DATE=199912
TRAIN_FILE=row-norm-all-100M.dat

BIN=~/work/dnn-quant/scripts

for K in 64 32 12 01
do
    $BIN/train_log_reg.py --config=${CONFIG_FILE} --train_datafile=${TRAIN_FILE} \
	--model_dir=chkpts-k${K} --num_unrollings=${K} > stdout-logreg-k${K}.txt 2> stderr-logreg-k${K}.txt &
done
