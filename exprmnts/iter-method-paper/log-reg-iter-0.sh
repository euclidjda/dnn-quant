#! /usr/bin/env bash

ROOT=$DNN_QUANT_ROOT
BIN=$ROOT/scripts
DATA_DIR=$ROOT/datasets
TRAIN_DIR=train-logreg
TRAIN_FILE=row-norm-all-100M.dat
CONFIG_FILE=log-reg.conf
CHKPTS_NAME=chkpts-${TRAIN_DIR}

# make training directory if it does not exist
mkdir -p ${TRAIN_DIR}

# Sets that can be done on seperate GPU's
# 2000-2003, 2004-2007, 2008-2011, 2012-2015

START_YEAR=1985
END_YEAR=1990
YEAR=$START_YEAR

while [ $YEAR -le $END_YEAR ]
do
    TEST_START=${YEAR}01
    TEST_END=${YEAR}12
    TEST_END_W_PAD=`expr ${YEAR} + 1`01
    TEST_PRE=`expr ${YEAR} - 6`06
    TRAIN_END=`expr ${YEAR} - 2`12

    echo "Training model on 197001 to ${TRAIN_END} for test set year of ${YEAR}"
    $BIN/train_log_reg.py --config=${CONFIG_FILE} --default_gpu=/gpu:${GPU} --train_datafile=${TRAIN_FILE} \
        --end_date=${TRAIN_END} --model_dir=${CHKPTS_NAME}-${TEST_START} > ${TRAIN_DIR}/stdout-${TEST_START}.txt

    echo "Creating test data set for ${TEST_START} to ${TEST_END} (Test pre is ${TEST_PRE})"
    $BIN/slice_data.pl $TEST_PRE $TEST_END_W_PAD < ${DATA_DIR}/${TRAIN_FILE} > ${TRAIN_DIR}/test-data-${TEST_START}.dat

    echo "Creating predictions file for period ${TEST_PRE} to ${TEST_END}"
    $BIN/classify_data.py --config=${CONFIG_FILE} --default_gpu=/gpu:${GPU} --model_dir=${CHKPTS_NAME}-${TEST_START}  --print_start=${TEST_START} --print_end=${TEST_END} \
        --data_dir=. --test_datafile=${TRAIN_DIR}/test-data-${TEST_START}.dat --output=${TRAIN_DIR}/preds-${TEST_START}.dat > ${TRAIN_DIR}/results-${TEST_START}.txt

    echo "Slicing predictions file ${TEST_START} to ${TEST_END}"
    $BIN/slice_data.pl $TEST_START $TEST_END < ${TRAIN_DIR}/preds-${TEST_START}.dat > ${TRAIN_DIR}/test-preds-${TEST_START}.dat

    YEAR=`expr $YEAR + 1`
done

# Now gen preds for training data
TRAIN_END=`expr ${START_YEAR} - 1`12
TRAIN_TAG=${START_YEAR}01

$BIN/slice_data.pl 197001 ${TRAIN_END} < $DATA_DIR/${TRAIN_FILE} > ${TRAIN_DIR}/train-data.dat

$BIN/classify_data.py --config=${CONFIG_FILE} --default_gpu=/gpu:${GPU} --model_dir=${CHKPTS_NAME}-${TRAIN_TAG} --print_start=197001 --print_end=${TRAIN_END} \
    --data_dir="." --test_datafile=${TRAIN_DIR}/train-data.dat --output=${TRAIN_DIR}/train-preds.dat > ${TRAIN_DIR}/results-train.txt
