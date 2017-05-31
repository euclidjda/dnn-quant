#!/bin/sh

BIN=~/work/euclid2/bin

###### Value-Factor Model (TODO: THIS SHOULD BE RE-DONE TO BE CONSISTENT WITH BELOW
######

#echo "Simulation value-factor-30nms"
#echo `date`
#$BIN/fundsim --config=params/value-factor-30nms.conf datasets/row-norm-sim-1B-2000-2015.dat | $BIN/json-stats.pl > output/value-factor-2000-2015-30nms.json

#echo "Simulation value-factor-30pct"
#echo `date`
#$BIN/fundsim --config=params/value-factor-30pct.conf datasets/row-norm-sim-1B-2000-2015.dat | $BIN/json-stats.pl > output/value-factor-2000-2015-30pct.json

#echo "Simulation value-factor-bottom-30pct"
#echo `date`
#$BIN/fundsim --config=params/value-factor-bottom-30pct.conf datasets/row-norm-sim-1B-2000-2015.dat | $BIN/json-stats.pl > output/value-factor-2000-2015-bottom-30pct.json

###### FACTORS
######

echo "Simulation factors-30nms"
echo `date`
$BIN/fundsim --config=params/model-30nms.conf datasets/factors-sim-1B-factors-2000-2015.dat | $BIN/json-stats.pl > output/factors-2000-2015-30nms.json

echo "Simulation factors-30pct"
echo `date`
$BIN/fundsim --config=params/model-30pct.conf datasets/factors-sim-1B-factors-2000-2015.dat | $BIN/json-stats.pl > output/factors-2000-2015-30pct.json

echo "Simulation factors-bottom-30pct"
echo `date`
$BIN/fundsim --config=params/model-bottom-30pct.conf datasets/factors-sim-1B-factors-2000-2015.dat | $BIN/json-stats.pl > output/factors-2000-2015-bottom-30pct.json

###### SFM
######

echo "Simulation sfm-30nms"
echo `date`
$BIN/fundsim --config=params/model-30nms.conf datasets/factors-sim-1B-sfm-2000-2015.dat | $BIN/json-stats.pl > output/sfm-2000-2015-30nms.json

echo "Simulation sfm-30pct"
echo `date`
$BIN/fundsim --config=params/model-30pct.conf datasets/factors-sim-1B-sfm-2000-2015.dat | $BIN/json-stats.pl > output/sfm-2000-2015-30pct.json

echo "Simulation sfm-bottom-30pct"
echo `date`
$BIN/fundsim --config=params/model-bottom-30pct.conf datasets/factors-sim-1B-sfm-2000-2015.dat | $BIN/json-stats.pl > output/sfm-2000-2015-bottom-30pct.json

###### MLP-FACTORS
######

echo "Simulation mlp-factors-30nms"
echo `date`
$BIN/fundsim --config=params/model-30nms.conf datasets/factors-sim-1B-mlp-factors-2000-2015.dat | $BIN/json-stats.pl > output/mlp-factors-2000-2015-30nms.json

echo "Simulation mlp-factors-30pct"
echo `date`
$BIN/fundsim --config=params/model-30pct.conf datasets/factors-sim-1B-mlp-factors-2000-2015.dat | $BIN/json-stats.pl > output/mlp-factors-2000-2015-30pct.json

echo "Simulation mlp-factors-bottom-30pct"
echo `date`
$BIN/fundsim --config=params/model-bottom-30pct.conf datasets/factors-sim-1B-mlp-factors-2000-2015.dat | $BIN/json-stats.pl > output/mlp-factors-2000-2015-bottom-30pct.json


