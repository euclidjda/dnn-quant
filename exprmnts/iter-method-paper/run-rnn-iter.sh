#! /usr/bin/env bash                                                                                                                                                                                                                 
                                                                                                                                                                                                                                               
# --model_dir           rnn-gru-it-chkpts                                                                                                                                                                                                              
# --train_datafile      train-1yr.dat
# --end_date            199812

ROOT=$DNN_QUANT_ROOT
BIN=$ROOT/scripts
DATA_DIR=$ROOT/datasets
TRAIN_DIR=train-data-rnn
#TRAIN_FILE=row-norm-all-100M.dat
TRAIN_FILE=all-1yr.dat
SIM_FILE=row-norm-all-1B.dat
CONFIG_FILE=rnn-gru-iter.conf

YEAR=1998
START_YEAR=YEAR

#while [ $YEAR -lt 2012 ]
while [ $YEAR -lt 1999 ]
do
    TRAIN_END=${YEAR}12
    TEST_PRE=`expr ${TRAIN_END:0:4} - 4`06
    TEST_START=`expr ${TRAIN_END:0:4} + 2`01
    TEST_END=`expr ${TRAIN_END:0:4} + 2`12

    # $BIN/train_net.py --config=${CONFIG_FILE} --train_datafile=${TRAIN_FILE} --end_date=${TRAIN_END} --model_dir=rnn-chkpts-${TEST_START} > ${TRAIN_DIR}/stdout-${TEST_START}.txt

    echo "Training ends on ${TRAIN_END}"
    echo "Date range for testing ${TEST_PRE} to ${TEST_END}"
    echo "Date range for saved estimates ${TEST_START} to ${TEST_END}"

    $BIN/slice_data.pl $TEST_PRE $TEST_END < ${DATA_DIR}/${TRAIN_FILE} > ${TRAIN_DIR}/test-data-${TEST_START}.dat

    $BIN/classify_data.py --config=${CONFIG_FILE} --model_dir=rnn-chkpts-${TEST_START}  --print_start=${TEST_START} --print_end=${TEST_END} \
        --data_dir=. --test_datafile=${TRAIN_DIR}/test-data-${TEST_START}.dat --output=${TRAIN_DIR}/preds-${TEST_START}.dat > ${TRAIN_DIR}/results-${TEST_START}.txt

    $BIN/slice_data.pl $TEST_START $TEST_END < ${TRAIN_DIR}/preds-${TEST_START}.dat > ${TRAIN_DIR}/test-preds-${TEST_START}.dat

    YEAR=`expr $YEAR + 1`
done

# Now gen preds for training data

#$BIN/slice_data.pl 197001 199912 < $DATA_DIR/${TRAIN_FILE} > ${TRAIN_DIR}/train-data.dat

#$BIN/classify_data.py --config=${CONFIG_FILE} --model_dir=rnn-chkpts-200001 \
#    --data_dir=. --test_datafile=${DATA_DIR}/train-data.dat --output=${TRAIN_DIR}/train-preds.dat > ${TRAIN_DIR}/results-train.txt

#cat ${TRAIN_DIR}/train-preds.dat ${TRAIN_DIR}/test-preds-*.dat > ${TRAIN_DIR}/preds-all.dat
#$BIN/merge_model_with_simdata.pl ${TRAIN_DIR}/preds-all.dat $DATA_DIR/${TRAIN_FILE} > ${TRAIN_DIR}/rnn-sim-data-100M.dat
#$BIN/merge_model_with_simdata.pl ${TRAIN_DIR}/preds-all.dat $DATA_DIR/${SIM_FILE} > ${TRAIN_DIR}/rnn-sim-data-1B.dat
