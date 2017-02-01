#! /usr/bin/env perl

my $MIN = 0.000001;
my $MAX = 1.000000;

use strict;

my $total = 0;
my $count = 0;

while(<STDIN>) {

    chomp;
    my @fields = split ' ';

    my $target = $fields[4];
    my $p0 = $fields[2];
    my $p1 = $fields[3];

    my $term = 0.0;

    $term = $target ? $p1 : $p0;

    $term = $MIN if $term < $MIN;
    $term = $MAX if $term > $MAX;

    $total += -log( $term );
    $count++;

    #printf("%.4f %.4f %d\n",$total,$term,$count);
    #die if $count > 10;
}

my $logloss = exp($total/$count);

printf("%.8f\n",$logloss);

