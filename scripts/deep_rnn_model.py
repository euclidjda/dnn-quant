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
from tensorflow.models.rnn import rnn

class DeepRnnModel(object):
  """The deep rnn model."""

  def __init__(self, training, config):

      self._training = training
      self._batch_size = batch_size = config.batch_size
      self._num_unrollings = num_unrollings = config.num_unrollings
      num_hidden  = config.num_hidden
      num_inputs  = config.num_inputs
      num_outputs = config.num_outputs

      self._seq_lengths = tf.placeholder(tf.int32, [batch_size])
    
      self._inputs = list()
      self._targets = list()

      for _ in range(num_unrollings):
        self._inputs.append( tf.placeholder(tf.float32,shape=[batch_size,num_inputs]) )

      for _ in range(num_unrollings):
        self._targets.append( tf.placeholder(tf.float32,shape=[batch_size,num_outputs]) )

      # GRUCell alternative: lstm_cell = tf.nn.rnn_cell.GRUCell(num_hidden)
      lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden,forget_bias=0.0)

      if training and config.keep_prob < 1:
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
      
      cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

      self._state_size = cell.state_size
      # print("State size is: %d"%self._state_size)
      
      self._reset_state_flags = tf.placeholder(tf.float32,shape=[batch_size,cell.state_size])
      self._saved_state = tf.Variable(tf.zeros([batch_size, cell.state_size]), dtype=tf.float32,
                                      trainable=False)
      self._reset_state = self._saved_state.assign( tf.zeros([batch_size, cell.state_size] ) )  
    
      state = tf.mul( self._saved_state, self._reset_state_flags )
    
      outputs, state = rnn.rnn(cell, self._inputs,
                               initial_state=state, sequence_length=self._seq_lengths)

      softmax_w = tf.get_variable("softmax_w", [num_hidden, num_outputs])
      softmax_b = tf.get_variable("softmax_b", [num_outputs])

      with tf.control_dependencies([self._saved_state.assign(state)]):
        logits = tf.nn.xw_plus_b( tf.concat(0, outputs), softmax_w, softmax_b)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits, tf.concat(0, self._targets))

      #iters = batch_size*num_unrollings
      iters = tf.reduce_sum( tf.to_float( self._seq_lengths ) )

      self._cost = cost = tf.reduce_sum(loss) / iters
      self._final_state = state
      self._predictions = tf.nn.softmax(logits)
      self._error = 1.0 - tf.reduce_sum( tf.mul( tf.floor( self._predictions + 0.5 ),
                                                   tf.concat(0, self._targets) ) ) / iters
    
      if not training:
        return

      self._lr = tf.Variable(0.0, trainable=False)
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),config.max_grad_norm)
      optimizer = tf.train.GradientDescentOptimizer(self.lr)
      self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def step(self, sess,eval_op, x_batches, y_batches, seq_lengths, reset_flags):
    
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
                                                  eval_op],
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
  def train_op(self):
    return self._train_op

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def num_unrollings(self):
    return self._num_unrollings
