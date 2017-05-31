#!/bin/bash

BIN=$DNN_QUANT_ROOT/scripts

for AUM in 100M 400M 1B
do
    for NNTYPE in mlp rnn
    do
	for CTYPE in c03 c10
	do
	    echo "Building sim files for AUM=$AUM, NNTYPE=$NNTYPE, and CTYPE=$CTYPE"
	    $BIN/merge_model_with_simdata.pl datasets/row-norm-sim-$AUM.dat datasets/preds-$NNTYPE-$CTYPE.dat > datasets/row-norm-sim-$AUM-$NNTYPE-$CTYPE.dat
	    $BIN/split.pl 200001 datasets/row-norm-sim-$AUM-$NNTYPE-$CTYPE-1970-1999.dat datasets/row-norm-sim-$AUM-$NNTYPE-$CTYPE-2000-2015.dat < datasets/row-norm-sim-$AUM-$NNTYPE-$CTYPE.dat
	done
    done
done

