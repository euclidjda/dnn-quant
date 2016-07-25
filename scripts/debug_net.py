# Copyright 2015 Google Inc. All Rights Reserved.
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

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile
from batch_generator import BatchGenerator
from deep_rnn_model import DeepRnnModel

flags = tf.flags
flags.DEFINE_string("default_gpu",'/gpu:0',"The default GPU to use for this process")
flags.DEFINE_string("train_datafile", 'datasets/dev-small-1yr.dat', "The training dataset file")
flags.DEFINE_string("valid_datafile", 'datasets/dev-small-1yr.dat', "The validation dataset file")
flags.DEFINE_string("model_dir", 'dev-001', "Directory for reading and writing model")
flags.DEFINE_float("lr_decay",0.9,"Learning rate decay")
flags.DEFINE_float("init_scale",0.1,"Initial scale for weights")
flags.DEFINE_float("max_grad_norm",5.0,"Bound on gradient to prevent exploding")
flags.DEFINE_float("learning_rate",1.0,"Initial learning rate")
flags.DEFINE_float("keep_prob",0.5,"Keep probability for dropout")
flags.DEFINE_integer("passes",1,"For each epoch, how many passes through data set")
flags.DEFINE_integer("max_epoch",200,"Stop after max_epochs")
flags.DEFINE_integer("min_history",36,"Minimum history required for validation and test")
flags.DEFINE_integer("num_unrollings",36,"Number of unrolling steps for RNN")
flags.DEFINE_integer("batch_size",4,"Size of each batch")
flags.DEFINE_integer("num_layers",2,"Numer of RNN layers")
flags.DEFINE_integer("num_inputs",84,"Number of inputs")
flags.DEFINE_integer("hidden_size",128,"Number of hidden layer units")
flags.DEFINE_integer("num_outputs",2,"Number of classes for output")

FLAGS = flags.FLAGS

DATA_FILE_FMT = { 'key_name'        : 'gvkey'     ,
                  'date_name'       : 'date'      ,
                  'yval_name'       : 'target'    ,
                  'first_xval_name' : 'rel:mom1m' }

def run_epoch(session, mdl, batches, eval_op, passes=1, eval_all=True, verbose=False):
  """Runs the model on the given data."""
  num_steps = batches.num_steps
  start_time = time.time()
  costs = 0.0
  errors = 0.0
  count = 0
  dot_count = 0
  prog_int = passes*num_steps/100 # progress interval for stdout
  predata = False
  assert(num_steps > 0)
  # print("num_steps = ",num_steps)
  # print("data_points = ",batches.num_data_points())
  # print("batch_size = ",mdl.batch_size)
  # print("unrollings = ",mdl.num_unrollings)

  batches.rewind_cursor()

  for _ in range(passes):

    for step in range(num_steps):

      #if eval_all is False:
      #  predata = batches.is_predata(FLAGS.min_history)
      if (step == 0):
        print(batches._cursor)

      x_batches, y_batches, seq_lengths, reset_flags = batches.next()

      if (step == 0):
        print(reset_flags)
      
      cost, error, state, predictions = mdl.step( session, eval_op,
                                                    x_batches, y_batches,
                                                    seq_lengths, reset_flags )
      if not predata:
        costs  += cost
        errors += error
        count  += 1
      if ( verbose and ((prog_int<=1) or (step % (int(prog_int)+1)) == 0) ):
        dot_count += 1
        print('.',end='')
        sys.stdout.flush()

  if verbose:
    print("."*(100-dot_count),end='')
    print(" evals: %d (of %d), speed: %.0f seconds"%
            (count * mdl.batch_size * mdl.num_unrollings,
               passes * num_steps * mdl.batch_size * mdl.num_unrollings,
                  (time.time() - start_time) ) )
  sys.stdout.flush()

  return np.exp(costs / count), (errors / count)

def adjust_learning_rate(session, model, learning_rate, batch_perfs):
  if len(batch_perfs) > 5:
    mean = sum(batch_perfs[-5:-1])/4.0
    curr = batch_perfs[-1]
    # If performance has dropped by less than 1%, decay learning_rate
    if ((learning_rate >= 0.001) and (mean > 0.0) and (mean >= curr) and (curr/mean >= 0.99)):
        learning_rate = learning_rate * FLAGS.lr_decay
  model.assign_lr(session, learning_rate)
  return learning_rate
          
def main(_):

  config = FLAGS

  train_batches = BatchGenerator(config.train_datafile, DATA_FILE_FMT,
                                   config.num_inputs, config.num_outputs,
                                   config.batch_size, config.num_unrollings )

  valid_batches = BatchGenerator(config.valid_datafile, DATA_FILE_FMT,
                                   config.num_inputs, config.num_outputs,
                                   config.batch_size, config.num_unrollings  )
  
  tf_config = tf.ConfigProto( allow_soft_placement=True, log_device_placement=False )

  with tf.Graph().as_default(), tf.Session(config=tf_config) as session:
    initer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)

    with tf.variable_scope("model", reuse=None, initializer=initer), tf.device(config.default_gpu):
      mtrain = DeepRnnModel(training=False, config=config)
    with tf.variable_scope("model", reuse=True, initializer=initer), tf.device(config.default_gpu):
      mvalid = DeepRnnModel(training=False, config=config)

    saver = tf.train.Saver(max_to_keep=200)

    ckpt = tf.train.get_checkpoint_state(config.model_dir)
    if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
      print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      saver.restore(session, ckpt.model_checkpoint_path)
    else:
      print("Created model with fresh parameters.")
      session.run(tf.initialize_all_variables())

    valid_xentrop, valid_error = run_epoch(session,
                                             mvalid, valid_batches,
                                             tf.no_op(),
                                             passes=1,
                                             eval_all=False,
                                             verbose=True)

    check_xentrop, check_error = run_epoch(session,
                                             mtrain, train_batches,
                                             tf.no_op(),
                                             passes=1,
                                             eval_all=False,
                                             verbose=True)
      
    print("Check XEntrop: %.2f =? %.2f Error: %.6f =? %.6f " %
            (check_xentrop, valid_xentrop, check_error, valid_error))
    sys.stdout.flush()

if __name__ == "__main__":
  tf.app.run()
