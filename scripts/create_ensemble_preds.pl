#! /usr/bin/env perl

use strict;

my $PRED_FILE_GVKEY_IDX    = 0;
my $PRED_FILE_DATE_IDX     = 1;
my $PRED_FILE_NEG_PROB_IDX = 2;
my $PRED_FILE_POS_PROB_IDX = 3;
my $PRED_FILE_TARGET_IDX   = 4;
my $PRED_FILE_STEPS_IDX    = 5;
my $PRED_FILE_EXT_PROB_IDX = 6;

main();

sub main {

    $| = 1;

    my $method   = $ARGV[0] || "mean";

    while(<STDIN>) {

	chomp;
	my @fields = split(' ',$_);

	my $gvkey = $fields[$PRED_FILE_GVKEY_IDX];
	my $date  = $fields[$PRED_FILE_DATE_IDX];
	my $m1_prob = $fields[$PRED_FILE_POS_PROB_IDX];
	my $m2_prob = $fields[$PRED_FILE_EXT_PROB_IDX];

	defined($m1_prob) || die;
	defined($m2_prob) || die;

	my $ensemble_prob = undef;
	
	if ($method eq 'mean') {

	    $ensemble_prob = ($m1_prob+$m2_prob)/2.0;

	} elsif ($method eq 'min') {
	    
	    $ensemble_prob = $m1_prob > $m2_prob ? $m2_prob : $m1_prob;

	} elsif ($method eq 'max') {
	    
	    $ensemble_prob = $m1_prob > $m2_prob ? $m1_prob : $m2_prob;

	} elsif ($method eq 'conf') {
	    
	    if (abs($m1_prob - 0.5) > abs($m2_prob - 0.5)) {
		
		$ensemble_prob = $m1_prob;

	    } else {

		$ensemble_prob = $m2_prob;

	    }

	}


	printf("%s %s %.4f %.4f %d %d\n",
	       $gvkey,$date,
	       1.0-$ensemble_prob,
	       $ensemble_prob,
	       $fields[$PRED_FILE_TARGET_IDX],
	       $fields[$PRED_FILE_STEPS_IDX]);

    }

}
