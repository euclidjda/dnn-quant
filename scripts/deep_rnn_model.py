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

import os
import sys

import numpy as np
import tensorflow as tf

# from tensorflow.python.ops import array_ops

_NUM_OUTPUTS = 2

class DeepRnnModel(object):
  """
  A Deep Rnn Model that supports a binary (two class) output with an
  arbitrary number of fixed width hidden layers.
  """
  def __init__(self, num_layers, num_inputs, num_hidden,
               num_unrollings, batch_size,
               max_grad_norm=5.0, 
               use_fixed_k=False,
               input_dropout=False,optimizer='gd'):
      """
      Initialize the model
      Args:
        num_layers: number of hidden layers
        num_inputs: number input units. this should be less than or
          or equal to width of feature data in the data file
        num_hidden: number of hidden units in each hidden layer
        num_unrollings: the size of the time window processed in 
          each step (see step() function below)
        batch_size: the size of the data batch processed in each step
        max_grad_norm: max gardient norm size for gradient clipping
      """
      self._batch_size = batch_size
      self._num_unrollings = num_unrollings
      num_outputs = _NUM_OUTPUTS
      
      self._seq_lengths = tf.placeholder(tf.int32, shape=[batch_size])
      self._keep_prob = tf.placeholder(tf.float32, shape=[])

      self._inputs = list()
      self._targets = list()
      self._train_wghts = list() # Weights for loss functions per example
      self._valid_wghts = list() # Weights for loss functions per example

      for _ in range(num_unrollings):
        self._inputs.append( tf.placeholder(tf.float32,
                                              shape=[batch_size,num_inputs]) )
        self._targets.append( tf.placeholder(tf.float32,
                                              shape=[batch_size,num_outputs]) )
        self._train_wghts.append(tf.placeholder(tf.float32, shape=[batch_size]))
        self._valid_wghts.append(tf.placeholder(tf.float32, shape=[batch_size]))
        
      rnn_cell = tf.nn.rnn_cell.GRUCell(num_hidden)
      #rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden,
      #                                             forget_bias=0.0)

      input_keep_prob = self._keep_prob if input_dropout is True else 1.0

      rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, 
                                               output_keep_prob=self._keep_prob, 
                                               input_keep_prob=input_keep_prob)
      
      cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * num_layers)

      self._state_size = cell.state_size

      state_shape=[batch_size, cell.state_size]
      self._reset_state_flags = tf.placeholder(tf.float32, shape=state_shape)
      self._saved_state = tf.Variable(tf.zeros(state_shape), dtype=tf.float32,
                                          trainable=False)

      init_state = None

      if use_fixed_k is False:
        init_state = tf.mul( self._saved_state, self._reset_state_flags )
    
      outputs, state = tf.nn.rnn(cell, self._inputs,
                                 initial_state=init_state,
                                 dtype=tf.float32,
                                 sequence_length=self._seq_lengths)

      self._state = state

      softmax_w = tf.get_variable("softmax_w", [num_hidden+num_inputs*num_unrollings, 
                                                num_outputs])
      softmax_b = tf.get_variable("softmax_b", [num_outputs])

      skip_con_inputs = list()
      for _ in range(num_unrollings-1):
        skip_con_inputs.append(tf.Variable(tf.zeros([batch_size,num_unrollings*num_inputs]),
                                dtype=tf.float32, trainable=False))
      skip_con_inputs.append( tf.concat( 1, self._inputs ) )

      with tf.control_dependencies([self._saved_state.assign(state)]):
        logits = tf.nn.xw_plus_b( tf.concat(1, [ tf.concat(0, skip_con_inputs), 
                                                 tf.concat(0, outputs)]),
                                  softmax_w, softmax_b)

      targets = tf.concat(0, self._targets)

      agg_loss = tf.nn.softmax_cross_entropy_with_logits(logits,targets)

      train_wghts = tf.concat(0, self._train_wghts)
      valid_wghts = tf.concat(0, self._valid_wghts)

      train_loss = tf.mul(agg_loss, train_wghts)
      valid_loss = tf.mul(agg_loss, valid_wghts)
            
      self._loss = self._train_loss = train_loss
      self._valid_loss = valid_loss

      self._train_evals = tf.reduce_sum( train_wghts )
      self._valid_evals = tf.reduce_sum( valid_wghts )

      self._train_cst = tf.reduce_sum( train_loss )
      self._valid_cst = tf.reduce_sum( valid_loss )

      self._predictions = tf.nn.softmax(logits)
      class_predictions = tf.floor( self._predictions + 0.5 )

      accy = tf.mul(class_predictions, targets)

      train_accy = tf.mul(accy,tf.reshape(train_wghts,
                                          shape=[batch_size*num_unrollings,1]))
      valid_accy = tf.mul(accy,tf.reshape(valid_wghts,
                                          shape=[batch_size*num_unrollings,1]))

      self._train_accy = tf.reduce_sum( train_accy )
      self._valid_accy = tf.reduce_sum( valid_accy )

      self._cost  = self._train_cst
      self._accy  = self._train_accy
      self._evals = self._train_evals
      self._batch_cst = self._train_cst / (self._train_evals + 1.0)

      # here is the learning part of the graph
      
      self.lr = tf.Variable(0.0, trainable=False)
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(self._batch_cst,
                                                         tvars),max_grad_norm)
      if optimizer == 'gd':
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
      elif optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(self.lr)
      elif optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(self.lr)
      elif optimizer == 'mo':
        optimizer = tf.train.MomentumOptimizer(self.lr)

      self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def train_step(self, sess, batch, keep_prob=1.0):
    """
    Take one step through the data set. A step contains a sequences of batches
    where the sequence is of size num_unrollings. The batches are size
    batch_size. 
    Args:
      sess: current tf session the model is being run in
      batch: batch of data of type Batch (see batch_generator.py)
      keep_prob: keep_prob for dropout
    Returns:
      train_cost: cross entropy cost function for the next batch in batches
      train_accy: binary classifcation accuracy for the next batch in batches
      train_evals:
      valid_cost: 
      valid_accy:
      valid_evals:
    """

    feed_dict = self._get_feed_dict(batch,keep_prob)

    (train_cst,train_accy, train_evals,
     valid_cst, valid_accy, valid_evals,
     _) = sess.run([self._train_cst,
                    self._train_accy,
                    self._train_evals,
                    self._valid_cst,
                    self._valid_accy,
                    self._valid_evals,
                    self._train_op],
                    feed_dict)

    return (train_cst, train_accy, train_evals, 
            valid_cst, valid_accy, valid_evals)

  def step(self, sess, batch):
     """
     Take one step through the data set. A step contains a sequences of batches
     where the sequence is of size num_unrollings. The batches are size
     batch_size. 
     Args:
       sess: current tf session the model is being run in
       batch: batch of data of type Batch
     Returns:
       predictions: the model predictions for each data point in batch
     """
     assert( len(batch.inputs) ==self._num_unrollings )
      
     feed_dict = self._get_feed_dict(batch)

     predictions = sess.run(self._predictions,feed_dict)

     return predictions

   
  def _get_feed_dict(self,batch,keep_prob=1.0):

    print("In deep_rnn_model.py: ")
    print(batch.seq_lengths)
    print("")

    reset_flags = np.repeat( batch.reset_flags.reshape( [self._batch_size, 1] ),
                               self._state_size, axis=1 )

    feed_dict = dict()

    feed_dict[self._keep_prob] = keep_prob
    feed_dict[self._reset_state_flags] = reset_flags
    feed_dict[self._seq_lengths] = batch.seq_lengths
    
    for i in range(self._num_unrollings):
      feed_dict[self._inputs[i]]  = batch.inputs[i]
      feed_dict[self._targets[i]] = batch.targets[i]
      feed_dict[self._train_wghts[i]] = batch.train_wghts[i]
      feed_dict[self._valid_wghts[i]] = batch.valid_wghts[i]
    
    return feed_dict

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def inputs(self):
    return self._inputs

  @property
  def targets(self):
    return self._targets

  @property
  def cost(self):
    return self._cost

  @property
  def num_unrollings(self):
    return self._num_unrollings
