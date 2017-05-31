#!/bin/sh

BIN=~/work/euclid2/bin

for PERIOD in "1970-1999" "2000-2015"
do
    for MRKCAP in 100M 400M 1B
    do
	for CTYPE in c10 c03
	do
	    for NNTYPE in mlp rnn
	    do
		echo "Simulation all for MRKCAP=$MRKCAP NNTYPE=$NNTYPE CTYPE=$CTYPE for period $PERIOD"
		echo `date`
		echo "50 Names"
		$BIN/fundsim --config=params/model-50nms.conf datasets/row-norm-sim-$MRKCAP-$NNTYPE-$CTYPE-$PERIOD.dat | $BIN/json-stats.pl > output/output-$MRKCAP-$NNTYPE-$CTYPE-$PERIOD-50nms.dat
		
		echo "Top 30pct"
		echo `date`
		$BIN/fundsim --config=params/model-30pct.conf datasets/row-norm-sim-$MRKCAP-$NNTYPE-$CTYPE-$PERIOD.dat | $BIN/json-stats.pl > output/output-$MRKCAP-$NNTYPE-$CTYPE-$PERIOD-30pct.dat
		
		echo "Bottom 30pct"
		echo `date`
		$BIN/fundsim --config=params/model-bottom-30pct.conf datasets/row-norm-sim-$MRKCAP-$NNTYPE-$CTYPE-$PERIOD.dat | $BIN/json-stats.pl > output/output-$MRKCAP-$NNTYPE-$CTYPE-$PERIOD-bottom-30pct.dat
		
	    done
	done
    done
done