#!/bin/sh

BIN=$DNN_QUANT_ROOT/scripts

cut -d ' ' -f 1-12 datasets/row-norm-sim-1B-2000-2015.dat > datasets/row-norm-source-2000-2015.dat

$BIN/merge_model_with_simdata.pl datasets/row-norm-source-2000-2015.dat datasets/ensemble-mean-preds-2000-2015.dat > datasets/row-norm-sim-1B-ensemble-mean-2000-2015.dat

$BIN/merge_model_with_simdata.pl datasets/row-norm-source-2000-2015.dat datasets/ensemble-min-preds-2000-2015.dat > datasets/row-norm-sim-1B-ensemble-min-2000-2015.dat

$BIN/merge_model_with_simdata.pl datasets/row-norm-source-2000-2015.dat datasets/ensemble-max-preds-2000-2015.dat > datasets/row-norm-sim-1B-ensemble-max-2000-2015.dat

$BIN/merge_model_with_simdata.pl datasets/row-norm-source-2000-2015.dat datasets/ensemble-conf-preds-2000-2015.dat > datasets/row-norm-sim-1B-ensemble-conf-2000-2015.dat
