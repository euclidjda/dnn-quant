#!/bin/sh
''''exec python3 -u -- "$0" ${1+"$@"} # '''

# #! /usr/bin/env python3
# Copyright 2016 Euclidean Technologies Management LLC All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import sys
import copy

import numpy as np
import tensorflow as tf

import model_utils
import configs

from tensorflow.python.platform import gfile
from batch_generator import BatchGenerator

def run_epoch(session, model, dataset,
                keep_prob=1.0, passes=1, verbose=False):
  """Run the specified data set through the model.

  Args:
    session: The tf session to run graph in
    model: The model. An object of type deep_rnn_model
    dataset: The data. An object of type BatchGenerator
    keep_prob: Dropout keep_prob for training
    verbose: Display iteration output to stdout
  Returns:
    train_cost: average cross-entropy loss value on data
    train_accy: average binary classification accuracy rate
    valid_cost: average cross-entropy loss value on data
    valid_accy: average binary classification accuracy rate
  Raises:
    RuntimeError: the batch size cannot be larger than the training
      data set size
  """
  num_batches = dataset.num_batches
  start_time = time.time()
  train_cost = train_accy = valid_cost = valid_accy = 0.0
  train_evals = valid_evals = 0.0
  dot_count = 0
  count = passes*num_batches
  prog_int = count/100 # progress interval for stdout

  if not num_batches > 0:
    raise RuntimeError("batch_size*num_unrollings is larger "
                         "than the training set size.")

  dataset.rewind() # make sure we start a beggining

  print("batches: %d "%num_batches,end=' ')

  for i in range(passes):
    for step in range(num_batches):
      batch = dataset.next_batch()
      (tcost, taccy, tevals,
       vcost, vaccy, vevals) = model.train_step(session, batch,
                                               keep_prob=keep_prob)

      train_cost  += tcost
      train_accy  += taccy
      train_evals += tevals
      valid_cost  += vcost
      valid_accy  += vaccy
      valid_evals += vevals

      #print("-"*80)
      #print("train_evals %d"%train_evals)
      #print("valid_evals %d"%valid_evals)
      #print("train_cost %.2f"%train_cost)
      #print("train_accy %.2f"%train_accy)
      #print("-"*80)
      # exit()
      if ( verbose and ((prog_int<=1) or
                        (step % (int(prog_int)+1)) == 0) ):
        dot_count += 1
        print('.',end='')
        sys.stdout.flush()
      #exit()
  #exit()
  if verbose:
    print("."*(100-dot_count),end='')
    print(" passes: %d train iters: %d valid iters: %d "
          "speed: %.0f seconds" % (passes,
                                   train_evals,
                                   valid_evals,
                                   (time.time() - start_time)) )
  sys.stdout.flush()

  return (np.exp(train_cost/train_evals),
          1.0 - train_accy/train_evals,
          np.exp(valid_cost/valid_evals),
          1.0 - valid_accy/valid_evals)

def main(_):
  """
  Entry point and main loop for train_net.py. Uses command line arguments to get
  model and training specification (see config.py).
  """
  configs.DEFINE_string("train_datafile", None,"Training file")
  configs.DEFINE_float("lr_decay",0.9, "Learning rate decay")
  configs.DEFINE_float("initial_learning_rate",1.0,"Initial learning rate")
  configs.DEFINE_float("validation_size",0.0,"Size of validation set as %")
  configs.DEFINE_integer("passes",1,"Passes through day per epoch")
  configs.DEFINE_integer("max_epoch",0,"Stop after max_epochs")
  configs.DEFINE_integer("early_stop",None,"Early stop parameter")
  configs.DEFINE_integer("seed",None,"Seed for deterministic training")

  config = configs.get_configs()

  if config.train_datafile is None:
     config.train_datafile = config.datafile

  train_path = model_utils.get_data_path(config.data_dir,config.train_datafile)

  print("Loading training data ...")

  rand_samp = True if config.use_fixed_k is True else False

  train_data = BatchGenerator(train_path, config,
			      config.batch_size,config.num_unrollings,
			      validation_size=config.validation_size,
			      randomly_sample=rand_samp)

  tf_config = tf.ConfigProto( allow_soft_placement=True,
                              log_device_placement=False )

  with tf.Graph().as_default(), tf.Session(config=tf_config) as session:

    if config.seed is not None:
      tf.set_random_seed(config.seed)

    print("Constructing model ...")

    model = model_utils.get_training_model(session, config, verbose=True)

    lr = config.initial_learning_rate
    train_history = list()
    valid_history = list()

    for i in range(config.max_epoch):

      lr = model_utils.adjust_learning_rate(session,
                                              model, lr,
                                              config.lr_decay,
                                              train_history )

      trc, tre, vdc, vde = run_epoch(session, model, train_data,
                                       keep_prob=config.keep_prob,
                                       passes=config.passes,
                                       verbose=True)

      trc = 999.0 if trc > 999.0 else trc
      vdc = 999.0 if vdc > 999.0 else vdc

      print( ('Epoch: %d loss: %.6f %.6f'
              ' error: %.6f %.6f Learning rate: %.4f') %
            (i + 1, trc, vdc, tre, vde, lr) )
      sys.stdout.flush()

      train_history.append( trc )
      valid_history.append( vdc )

      if not os.path.exists(config.model_dir):
        print("Creating directory %s" % config.model_dir)
        os.mkdir(config.model_dir)

      chkpt_file_prefix = "training.ckpt"

      if model_utils.stop_training(config,valid_history,chkpt_file_prefix):
        print("Training stopped.")
        quit()
      else:
        checkpoint_path = os.path.join(config.model_dir, chkpt_file_prefix)
        tf.train.Saver().save(session, checkpoint_path, global_step=i)

if __name__ == "__main__":
  tf.app.run()
