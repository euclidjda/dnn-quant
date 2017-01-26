#! /usr/bin/env perl

use strict;

my $POS_PRED_IDX = 3;
my $TARGET_IDX   = 4;

my %date_to_preds = ();

while(<STDIN>) {

    chomp;
    my @fields = split ' ';

    my $date = $fields[1];

    $date_to_preds{$date} = [ ] unless exists($date_to_preds{$date});
    
    push @{$date_to_preds{$date}}, \@fields;

}

my @correct = ();

for ( 1 .. 10 ) {

    push @correct, [ ];

}

foreach my $date (sort keys %date_to_preds) {

    my $preds_ref = $date_to_preds{$date};

    my $idx = $POS_PRED_IDX;
    my @preds = sort { $b->[$idx] <=> $a->[$idx] } @$preds_ref;

    my $size = scalar(@correct)*10;
    my $num_correct = 0;

    foreach my $i ( 0 .. $size-1 ) {

	my $cur_pred = @preds[$i];

	$num_correct++ if $cur_pred->[$TARGET_IDX];
	#printf("%d\n",$num_correct);

	my $posnum = $i+1;

	if (not ($posnum % 10)) {
	
	    my $j = $posnum/10 - 1;
	    #print($i,' ',$posnum,' ',$idx,"\n");
	    push @{$correct[$j]}, sprintf("%.6f",$num_correct/$posnum);

	}

    }

}

for my $i ( 0 .. $#correct ) {

    my $posnum = ($i+1)*10;
    my $mean  = mean($correct[$i]);
    my $stdev = stdev($correct[$i]);
    my $ratio = $mean/$stdev;

    printf("%d %.4f %.4f\n",
	   $posnum,$mean,$stdev);

}

sub mean {

    my $values_ref = ref $_[0] ? shift(@_) : \@_;

    my $sum_of_values = 0;
    my $count = 0;

    foreach my $value (@$values_ref) {

	if ( defined $value ) {

	    $sum_of_values += $value;
	    $count++;
	}
    }

    return $count ? $sum_of_values/$count : undef;


}

sub stdev {

    my $values_ref = ref $_[0] ? shift(@_) : \@_;

    return undef unless @$values_ref;

    my $mean = mean($values_ref);
    my $sum = 0;
    my $count = 0;

    foreach my $value (@$values_ref) {

	if (defined($value)) {

	    $sum += ($value - $mean) * ($value - $mean);
	    $count++;

	}

    }

    my $stddev = undef;

    if ($count > 1) {

	$stddev = $sum / ($count - 1);

    } 

    return undef if !defined($stddev);

    return sqrt($stddev);

}
