#! /usr/bin/env python3

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

def run_epoch(session, model, dataset, passes=1, verbose=False):
  """Run the specified data set through the model.

  Args:
    session: The tf session to run graph in
    model: The model. An object of type deep_rnn_model
    dataset: The data. An object of type BatchGenerator
    passes: The number of times to run through the entire dataset
    verbose: Display iteration output to stdout
  Returns:
    cost: average cross-entropy loss value on data
    error: average binary classification error rate on data
  Raises:
    RuntimeError: the batch size cannot be larger than the training
      data set size
  """
  num_batches = dataset.num_batches
  start_time = time.time()
  costs = 0.0
  errors = 0.0
  count = 0
  dot_count = 0
  prog_int = passes*num_batches/100 # progress interval for stdout
  
  if not num_batches > 0:
    raise RuntimeError("batch_size*num_unrollings is larger "
                         "than the training set size.")

  dataset.rewind() # make sure we start a beggining

  for _ in range(passes):

    for step in range(num_batches):

      batch = dataset.next_batch()
      cost, error, _ = model.step(session, batch)
      costs  += cost
      errors += error
      count  += 1
      if ( verbose and ((prog_int<=1) or (step % (int(prog_int)+1)) == 0) ):
        dot_count += 1
        print('.',end='')
        sys.stdout.flush()

  if verbose:
    print("."*(100-dot_count),end='')
    print(" passes: %d itters: %d, speed: %.0f seconds"%
            (passes, count*model.batch_size, (time.time() - start_time) ) )
  sys.stdout.flush()

  return np.exp(costs / count), (errors / count)

def main(_):
  """
  Entry point and main loop for train_net.py. Uses command line arguments to get
  model and training specification (see config.py).
  """
  config = configs.get_configs()

  train_path = model_utils.get_data_path(config.data_dir,config.train_datafile)
  valid_path = model_utils.get_data_path(config.data_dir,config.valid_datafile)
 
  train_data = BatchGenerator(train_path,
                                   config.key_name, config.target_name,
                                   config.num_inputs,
                                   config.batch_size,
                                   config.num_unrollings )

  valid_data = BatchGenerator(valid_path,
                                   config.key_name, config.target_name,
                                   config.num_inputs,
                                   config.batch_size,
                                   config.num_unrollings )
  
  tf_config = tf.ConfigProto( allow_soft_placement=True, 
                              log_device_placement=False )

  with tf.Graph().as_default(), tf.Session(config=tf_config) as session:

    mtrain, mvalid = model_utils.get_training_models(session, config,
                                                       verbose=True)
    
    lr = config.initial_learning_rate
    perf_history = list()
    
    for i in range(config.max_epoch):

      lr = model_utils.adjust_learning_rate(session,
                                              mtrain,
                                              lr,
                                              config.lr_decay,
                                              perf_history )

      train_xentrop, train_error = run_epoch(session,
                                                  mtrain, train_data,
                                                  passes=config.passes,
                                                  verbose=True)

      valid_xentrop, valid_error = run_epoch(session,
                                                  mvalid, valid_data,
                                                  passes=1,
                                                  verbose=True)
      
      print( ('Epoch: %d XEntrop: %.6f %.6f'
              ' Error: %.6f %.6f Learning rate: %.3f') % 
            (i + 1, 
             train_xentrop, valid_xentrop, train_error, valid_error, lr) )
      sys.stdout.flush()

      perf_history.append( train_xentrop )

      if not os.path.exists(config.model_dir):
        print("Creating directory %s" % config.model_dir)
        os.mkdir(config.model_dir)
      
      checkpoint_path = os.path.join(config.model_dir, "training.ckpt" )
      tf.train.Saver().save(session, checkpoint_path, global_step=i)
      
      # If train and valid are the same data (e.g. as in the system-test)
      # then this is a test to make
      # sure that the model is producing the same error on both. Note,
      # however, if keep_prob < 1 then the train model is probabilistic
      # and so these can only be approx equal.
      if (False):
        check_xentrop, check_error = run_epoch(session,
                                                    mtrain, train_data,
                                                    passes=1,
                                                    verbose=True)
        print("Check: %d XEntrop: %.2f =? %.2f Error: %.6f =? %.6f " %
                (i + 1, check_xentrop, valid_xentrop, check_error, valid_error))
        sys.stdout.flush()

if __name__ == "__main__":
  tf.app.run()
