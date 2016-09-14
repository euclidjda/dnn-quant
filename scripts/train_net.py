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
    train_error: average binary classification error rate on data
    valid_cost: average cross-entropy loss value on data
    valid_error: average binary classification error rate on data
  Raises:
    RuntimeError: the batch size cannot be larger than the training
      data set size
  """
  num_batches = dataset.num_batches
  start_time = time.time()
  train_cst = train_err = valid_cst = valid_err = 0.0
  dot_count = 0
  count = passes*num_batches
  prog_int = count/100 # progress interval for stdout
  
  if not num_batches > 0:
    raise RuntimeError("batch_size*num_unrollings is larger "
                         "than the training set size.")

  dataset.rewind() # make sure we start a beggining

  for i in range(passes):
    for step in range(num_batches):
      batch = dataset.next_batch()
      tcst, terr, vcst, verr = model.train_step(session, batch,
                                                  keep_prob=keep_prob)

      train_cst += tcst
      train_err += terr
      valid_cst += vcst
      valid_err += verr
    
      if ( verbose and ((prog_int<=1) or (step % (int(prog_int)+1)) == 0) ):
        dot_count += 1
        print('.',end='')
        sys.stdout.flush()

  if verbose:
    print("."*(100-dot_count),end='')
    print(" passes: %d speed: %.0f seconds" % (passes,
                                                 (time.time() - start_time)) )
  sys.stdout.flush()

  train_cst = np.exp( train_cst / count )
  train_err = train_err / count
  vaild_cst = np.exp( valid_cst / count )
  valid_err = valid_err / count
  
  return train_cst, train_err, valid_cst, valid_err
  
def main(_):
  """
  Entry point and main loop for train_net.py. Uses command line arguments to get
  model and training specification (see config.py).
  """
  config = configs.get_configs()

  train_path = model_utils.get_data_path(config.data_dir,config.train_datafile)
  
  train_data = BatchGenerator(train_path,
                                   config.key_field,
                                   config.target_field,
                                   config.num_inputs,
                                   config.batch_size,
                                   config.num_unrollings )
  
  tf_config = tf.ConfigProto( allow_soft_placement=True, 
                              log_device_placement=False )

  with tf.Graph().as_default(), tf.Session(config=tf_config) as session:

    model = model_utils.get_training_model(session, config, verbose=True)
    
    lr = config.initial_learning_rate
    perf_history = list()
    
    for i in range(config.max_epoch):

      lr = model_utils.adjust_learning_rate(session,
                                              model, lr,
                                              config.lr_decay,
                                              perf_history )

      trc, tre, vdc, vde = run_epoch(session, model, train_data,
                                       keep_prob=1.0,
                                       passes=config.passes,
                                       verbose=True)
      print( ('Epoch: %d xentrpy: %.6f %.6f'
              ' error: %.6f %.6f Learning rate: %.3f') % 
            (i + 1, trc, vdc, tre, vde, lr) )
      sys.stdout.flush()

      perf_history.append( trc )

      if not os.path.exists(config.model_dir):
        print("Creating directory %s" % config.model_dir)
        os.mkdir(config.model_dir)
      
      checkpoint_path = os.path.join(config.model_dir, "training.ckpt" )
      tf.train.Saver().save(session, checkpoint_path, global_step=i)
      
if __name__ == "__main__":
  tf.app.run()
