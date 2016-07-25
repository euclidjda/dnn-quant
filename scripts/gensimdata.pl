#!/usr/local/bin/perl -w

use strict;

main();

sub main {

    $| = 1;

    my $datafile = $ARGV[0] || die;
    my $probfile = $ARGV[1] || die;

    my %probs = ();

    open(F1,"< $probfile");

    while(<F1>) {

	chomp;
	my @fields = split(' ',$_);

	next if $fields[0] eq 'gvkey'; # skip header

	my $gvkey = $fields[0];
	my $date  = $fields[1];
	my $prob  = $fields[4];
	$probs{"$gvkey$date"} = $prob;

    }

    close(F1);

    open(F2,"< $datafile");

    while(<F2>) {

	chomp;
	my @fields = split(' ',$_);

	my $gvkey = $fields[1];
	my $date  = $fields[0];
	my $prob = 'NULL';

	if (exists($probs{"$gvkey$date"})) {
	    $prob = $probs{"$gvkey$date"};
	}

	print join(' ',@fields[0..11]);

	print " $prob\n";

    }

    close(F2);



}
