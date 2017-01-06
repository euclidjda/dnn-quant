#! /usr/bin/env bash

START_YEAR=2000
END_YEAR=2015
YEAR=$START_YEAR

while [ $YEAR -le $END_YEAR ]
do

    grep error train-rnn/stdout-${YEAR}01.txt | cut -d ' ' -f 5,7,8 | sort | head -n 1 > train-rnn/valid-best-${YEAR}.txt
    
    YEAR=`expr $YEAR + 1`

done
