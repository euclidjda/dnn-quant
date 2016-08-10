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

from tensorflow.python.ops import array_ops
# from tensorflow.models.rnn import rnn

_NUM_OUTPUTS = 2

class DeepRnnModel(object):
  """
  A Deep Rnn Model that supports a binary (two class) output with an
  arbitrary number of fixed width hidden layers.
  """
  def __init__(self, num_layers, num_inputs, num_hidden,
                   num_unrollings, batch_size,
                   max_grad_norm=5.0, keep_prob=1.0, training=True):
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
        keep_prob: the keep probability for dropout training
        training: booling indicating whether training should be enabled
          for this model
      """
      self._training = training
      self._batch_size = batch_size = batch_size
      self._num_unrollings = num_unrollings = num_unrollings
      num_hidden  = num_hidden
      num_inputs  = num_inputs
      num_outputs = _NUM_OUTPUTS

      self._seq_lengths = tf.placeholder(tf.int32, [batch_size])
    
      self._inputs = list()
      self._targets = list()

      for _ in range(num_unrollings):
        self._inputs.append( tf.placeholder(tf.float32,
                                              shape=[batch_size,num_inputs]) )

      for _ in range(num_unrollings):
        self._targets.append( tf.placeholder(tf.float32,
                                               shape=[batch_size,num_outputs]) )

      rnn_cell = tf.nn.rnn_cell.GRUCell(num_hidden)
      #rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden,
      #                                             forget_bias=0.0)

      if training and keep_prob < 1:
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(
          rnn_cell, output_keep_prob=keep_prob)
      
      cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * num_layers)

      self._state_size = cell.state_size
      #print(self._state_size)
      #exit()
      state_shape=[batch_size, cell.state_size]
      self._reset_state_flags = tf.placeholder(tf.float32, shape=state_shape)
      self._saved_state = tf.Variable(tf.zeros(state_shape), dtype=tf.float32,
                                          trainable=False)
    
      state = tf.mul( self._saved_state, self._reset_state_flags )
    
      outputs, state = tf.nn.rnn(cell, self._inputs,
                                     initial_state=state,
                                     sequence_length=self._seq_lengths)

      softmax_w = tf.get_variable("softmax_w", [num_hidden, num_outputs])
      softmax_b = tf.get_variable("softmax_b", [num_outputs])

      with tf.control_dependencies([self._saved_state.assign(state)]):
        logits = tf.nn.xw_plus_b( tf.concat(0, outputs), softmax_w, softmax_b)
        targets = tf.concat(0, self._targets)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits, targets)

      iters = tf.reduce_sum( tf.to_float( self._seq_lengths ) )

      self._cost = cost = tf.reduce_sum(loss) / iters
      self._final_state = state
      self._predictions = tf.nn.softmax(logits)
      errors = tf.mul( tf.floor( self._predictions + 0.5 ), targets )
      self._error = 1.0 - tf.reduce_sum(errors) / iters
    
      if not training:
        self._train_op = tf.no_op()
        return

      self._lr = tf.Variable(0.0, trainable=False)
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),max_grad_norm)
      optimizer = tf.train.GradientDescentOptimizer(self.lr)
      self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def step(self, sess, batches):
    """
    Take one step through the data set. A step contains a sequences of batches
    where the sequence is of size num_unrollings. The batches are size
    batch_size. 
    Args:
      sess: current tf session the model is being run in
      batches: an obect of type BatchGenerator
    Returns:
      cost: cross entropy cost function for the next batch in batches
      error: binary classifcation error rate for the next batch in batches
      state: the final states of model after a step through the batch
      predictions: the model predictions for each data point in batch
    """
    x_batches, y_batches, seq_lengths, reset_flags = batches.next()
        
    feed_dict = dict()
      
    flags = np.repeat( reset_flags.reshape( [self._batch_size, 1] ),
                         self._state_size, axis=1 )

    feed_dict[self._reset_state_flags] =  flags
    feed_dict[self._seq_lengths] = seq_lengths
    
    for i in range(self._num_unrollings):
      feed_dict[self._inputs[i]]  = x_batches[i]
      feed_dict[self._targets[i]] = y_batches[i]

    cost, error, state, predictions, _ = sess.run([self._cost,
                                                  self._error,
                                                  self._final_state,
                                                  self._predictions,
                                                  self._train_op],
                                              feed_dict)

    return cost, error, state, predictions

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
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def num_unrollings(self):
    return self._num_unrollings
