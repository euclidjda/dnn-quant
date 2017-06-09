#! /usr/bin/env perl

my $MIN = 0.000001;
my $MAX = 1.000000;

use strict;

my $total = 0;
my $correct = 0;
my $count = 0;

while(<STDIN>) {

    chomp;
    my @fields = split ' ';

    my $target = int($fields[3]);
    my $p1     = $fields[2];
    my $p0     = 1.0 - $p1;

    my $term = 0.0;

    $term = $target ? $p1 : $p0;

    $term = $MIN if $term < $MIN;
    $term = $MAX if $term > $MAX;

    $total += -log( $term );
    $correct++ if ($target && ($p1>0.5)) || (!$target && ($p1<0.5));
    $count++;

    # printf("%.3f %.2f %.6f %.2f\n",$term,$target,$p1,$correct);
    # die if $count > 20;
}

my $logloss = $total/$count;
my $accy = $correct/$count;

printf("%.8f %.8f\n",1.0-$accy,$logloss);

