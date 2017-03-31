#!/usr/bin/perl -w

use strict;
use FileHandle;
use List::Util qw(shuffle);

my $OUTPUT_DIR = "output";
my $CONFIG_DIR = "config";
my $CHKPTS_DIR = "chkpts";

my @names = ();

for my $num_outputs (3,10) {

  for my $num_layers (1,2,4) {

      for my $num_hidden (64,256,512) {

	  for my $keep_prob (0.5,1.0) {

	      for my $init_rate (0.01,0.1,0.5,1.00,10.00) {

		  for my $init_scale (0.01,0.1,0.5) {

		      for my $max_norm (0,1,5,10,50,100) {

			  for my $loss_weight (0.50,0.75,1.00) {

			      my $name = sprintf("rnn-c%02d-l%d-h%03d-k%02d-i%04d-s%04d-m%03d-w%03d",
						 $num_outputs,
						 $num_layers,
						 $num_hidden,
						 int($keep_prob*10),
						 int($init_rate*100),
						 int($init_scale*100),
						 $max_norm,
						 int($loss_weight*100));

			      push(@names,$name);

my $CONFIG_STR =<<"CONFIG_STR";
--default_gpu		/gpu:0
--nn_type		rnn
--optimizer		adagrad
--key_field		gvkey
--target_field		target
--datafile		row-norm-all-100M.dat
--data_dir		datasets
--model_dir		$CHKPTS_DIR/chkpts-$name
--init_scale		$init_scale
--max_grad_norm		$max_norm
--initial_learning_rate $init_rate
--keep_prob		$keep_prob
--passes		0.3
--num_unrollings	48
--min_seq_length	24
--max_epoch		10000
--batch_size		256
--num_layers		$num_layers
--num_inputs		84
--num_hidden		$num_hidden
--num_outputs		$num_outputs
--validation_size	0.30
--seed			100
--early_stop		10
--end_date		199812
--rnn_loss_weight	$loss_weight
CONFIG_STR

my $fh= FileHandle->new("> $CONFIG_DIR/$name.conf");
			      $fh->autoflush(1);

    			      print $fh $CONFIG_STR;
			      close($fh);
			      
			  }

		      }

		  }

	      }
	  }
      }
  }

}

@names = shuffle(@names);

foreach my $name (@names) {

    printf("train_net.py --config=$CONFIG_DIR/$name.conf > $OUTPUT_DIR/stdout-$name.txt 2> $OUTPUT_DIR/stderr-$name.txt ; \n");

}

