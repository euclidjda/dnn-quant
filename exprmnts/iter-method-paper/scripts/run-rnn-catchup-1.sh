#! /usr/bin/env bash

ROOT=$DNN_QUANT_ROOT
BIN=$ROOT/scripts
DATA_DIR=$ROOT/datasets
TRAIN_DIR=train-rnn
TRAIN_FILE=row-norm-all-100M.dat
CONFIG_FILE=rnn-gru-iter.conf
CHKPTS_NAME=chkpts-${TRAIN_DIR}

GPU=1

MODEL=2003
YEAR=2005

MODEL_DATE=${MODEL}01
TEST_START=${YEAR}01
TEST_END=${YEAR}12
TEST_END_W_PAD=`expr ${YEAR} + 1`01
TEST_PRE=`expr ${YEAR} - 6`06
TRAIN_END=`expr ${YEAR} - 2`12

date
echo "Creating test data set for ${TEST_START} to ${TEST_END} (Test pre is ${TEST_PRE})"
$BIN/slice_data.pl $TEST_PRE $TEST_END_W_PAD < ${DATA_DIR}/${TRAIN_FILE} > ${TRAIN_DIR}/test-data-${TEST_START}.dat

date
echo "Creating predictions file for period ${TEST_PRE} to ${TEST_END}"
$BIN/classify_data.py --config=${CONFIG_FILE} --default_gpu=/gpu:${GPU} --model_dir=${CHKPTS_NAME}-${MODEL_DATE} --print_start=${TEST_START} --print_end=${TEST_END} \
    --data_dir=. --test_datafile=${TRAIN_DIR}/test-data-${TEST_START}.dat --output=${TRAIN_DIR}/catchup/preds-${TEST_START}.dat > ${TRAIN_DIR}/catchup/results-${TEST_START}.txt

date
echo "Slicing predictions file ${TEST_START} to ${TEST_END}"
$BIN/slice_data.pl $TEST_START $TEST_END < ${TRAIN_DIR}/catchup/preds-${TEST_START}.dat > ${TRAIN_DIR}/catchup/test-preds-${TEST_START}.dat


