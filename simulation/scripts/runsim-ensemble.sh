#!/bin/sh

BIN=~/work/euclid2/bin

for METHOD in mean min max conf
do
    echo "Simulation ensemble-${METHOD}-30nms"
    echo `date`
    $BIN/fundsim --config=params/model-30nms.conf datasets/row-norm-sim-1B-ensemble-${METHOD}-2000-2015.dat | $BIN/json-stats.pl > output/ensemble-${METHOD}-2000-2015-30nms.dat

    echo "Simulation ensemble-${METHOD}-30pct"
    echo `date`
    $BIN/fundsim --config=params/model-30pct.conf datasets/row-norm-sim-1B-ensemble-${METHOD}-2000-2015.dat | $BIN/json-stats.pl > output/ensemble-${METHOD}-2000-2015-30pct.dat

    echo "Simulation ensemble-${METHOD}-bottom-30pct"
    echo `date`
    $BIN/fundsim --config=params/model-bottom-30pct.conf datasets/row-norm-sim-1B-ensemble-${METHOD}-2000-2015.dat | $BIN/json-stats.pl > output/ensemble-${METHOD}-2000-2015-bottom-30pct.dat
done
