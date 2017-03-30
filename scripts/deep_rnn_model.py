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

from deep_nn_model import DeepNNModel

class DeepRnnModel(DeepNNModel):
  """
  A Deep Rnn Model that supports a binary (two class) output with an
  arbitrary number of fixed width hidden layers.
  """
  def __init__(self, num_layers, num_inputs, num_hidden, num_outputs,
               num_unrollings, batch_size,
               max_grad_norm=5.0, 
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
      
      self._seq_lengths = tf.placeholder(tf.int32, shape=[batch_size])
      self._keep_prob = tf.placeholder(tf.float32, shape=[])

      self._inputs = list()
      self._targets = list()
      self._train_mask = list()
      self._valid_mask = list()

      for _ in range(num_unrollings):
        self._inputs.append( tf.placeholder(tf.float32,
                                              shape=[batch_size,num_inputs]) )
        self._targets.append( tf.placeholder(tf.float32,
                                              shape=[batch_size,num_outputs]) )
        self._train_mask.append(tf.placeholder(tf.float32, shape=[batch_size]))
        self._valid_mask.append(tf.placeholder(tf.float32, shape=[batch_size]))
        
      rnn_cell = tf.contrib.rnn.GRUCell(num_hidden)
      #rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden,
      #                                             forget_bias=0.0)

      input_keep_prob = self._keep_prob if input_dropout is True else 1.0

      rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, 
                                               output_keep_prob=self._keep_prob, 
                                               input_keep_prob=input_keep_prob)
      
      rnn = tf.contrib.rnn.MultiRNNCell([rnn_cell] * num_layers)

      self._state_size = rnn.state_size

      state_shape=[batch_size, rnn.state_size]

      outputs, state = tf.contrib.rnn.static_rnn(rnn,
                                                 self._inputs,
                                                 dtype=tf.float32,
                                                 sequence_length=self._seq_lengths)

      self._state = state

      softmax_w = tf.get_variable("softmax_w", [num_hidden, num_outputs])
      softmax_b = tf.get_variable("softmax_b", [num_outputs])

      logits = tf.nn.xw_plus_b( tf.concat(outputs,0), softmax_w, softmax_b)

      targets = tf.concat(self._targets, 0)

      agg_loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets,logits=logits)

      train_mask = tf.concat(self._train_mask,0)
      valid_mask = tf.concat(self._valid_mask,0)

      train_loss = tf.multiply(agg_loss, train_mask)
      valid_loss = tf.multiply(agg_loss, valid_mask)
            
      self._loss = self._train_loss = train_loss
      self._valid_loss = valid_loss

      self._train_evals = tf.reduce_sum( train_mask )
      self._valid_evals = tf.reduce_sum( valid_mask )

      self._train_cst = tf.reduce_sum( train_loss )
      self._valid_cst = tf.reduce_sum( valid_loss )

      self._predictions = tf.nn.softmax(logits)
      # class_predictions = tf.floor( self._predictions + 0.5 )
      self._class_predictions = tf.one_hot(tf.argmax(self._predictions,1), 
                                           num_outputs, axis=-1)

      accy = tf.multiply(self._class_predictions, targets)

      train_accy = tf.multiply(accy,tf.reshape(train_mask,
                                          shape=[batch_size*num_unrollings,1]))
      valid_accy = tf.multiply(accy,tf.reshape(valid_mask,
                                          shape=[batch_size*num_unrollings,1]))

      self._train_accy = tf.reduce_sum( train_accy )
      self._valid_accy = tf.reduce_sum( valid_accy )

      self._cost  = self._train_cst
      self._accy  = self._train_accy
      self._evals = self._train_evals
      self._batch_cst = self._train_cst / (self._train_evals + 1.0)

      # here is the learning part of the graph
      tvars = tf.trainable_variables()
      grads = tf.gradients(self._batch_cst,tvars)

      if (max_grad_norm > 0):
        grads, self._grad_norm = tf.clip_by_global_norm(grads,max_grad_norm)
      else:
        self._grad_norm = tf.constant(0.0)

      self._lr = tf.Variable(0.0, trainable=False)
      optim = None
      if optimizer == 'gd':
        optim = tf.train.GradientDescentOptimizer(self._lr)
      elif optimizer == 'adagrad':
        optim = tf.train.AdagradOptimizer(self._lr)
      elif optimizer == 'adam':
        optim = tf.train.AdamOptimizer(self._lr)
      elif optimizer == 'mo':
        optim = tf.train.MomentumOptimizer(self._lr)
      else:
        raise RuntimeError("Unknown optimizer = %s"%optimizer)

      self._train_op = optim.apply_gradients(zip(grads, tvars))
