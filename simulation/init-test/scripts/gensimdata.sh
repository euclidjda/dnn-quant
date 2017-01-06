#!/bin/sh

ROOT=$DNN_QUANT_ROOT

cut -d ' ' -f 1-12 $ROOT/datasets/source-data-1B-1970-1999.dat > tmp.dat

$ROOT/scripts/merge_model_with_simdata.pl tmp.dat datasets/gvkey-date-pred.dat > datasets/simdata-1B-1970-1999.dat
