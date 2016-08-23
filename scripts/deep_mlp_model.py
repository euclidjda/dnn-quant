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

_NUM_OUTPUTS = 2

class DeepMlpModel(object):
  """
  A Deep MLP Model that supports a binary (two class) output with an
  arbitrary number of fixed width hidden layers.
  """
  def __init__(self, num_layers, num_inputs, num_hidden, batch_size,
                   max_grad_norm=5.0, keep_prob=1.0, training=True):
      """
      Initialize the model
      Args:
        num_layers: number of hidden layers
        num_inputs: number input units. this should be less than or
          or equal to width of feature data in the data file
        num_hidden: number of hidden units in each hidden layer
        batch_size: the size of the data batch processed in each step
        max_grad_norm: max gardient norm size for gradient clipping
        keep_prob: the keep probability for dropout training
        training: booling indicating whether training should be enabled
          for this model
      """
      self._training = training
      self._batch_size = batch_size
      num_outputs = _NUM_OUTPUTS

      self._inputs = tf.placeholder(tf.float32, shape=[batch_size,num_inputs])
      self._targets = tf.placeholder(tf.float32, shape=[batch_size,num_outputs])

      outputs = self._inputs
      num_prev = num_inputs

      for i in range(num_layers):
        weights = tf.get_variable("hidden_w_%d"%i,[num_prev, num_hidden])
        biases = tf.get_variable("hidden_b_%d"%i,[num_hidden])
        outputs = tf.nn.tanh(tf.nn.xw_plus_b(outputs, weights, biases))
        if training and keep_prob < 1.0:
          outputs = tf.nn.dropout(outputs,keep_prob)
        num_prev = num_hidden
        
      softmax_w = tf.get_variable("softmax_w", [num_prev, num_outputs])
      softmax_b = tf.get_variable("softmax_b", [num_outputs])

      targets = self._targets
      logits = tf.nn.xw_plus_b(outputs, softmax_w, softmax_b)
      loss = tf.nn.softmax_cross_entropy_with_logits(logits, targets)

      self._cost = cost = tf.reduce_sum(loss) / batch_size
      self._predictions = tf.nn.softmax(logits)
      errors = tf.mul( tf.floor( self._predictions + 0.5 ), targets )
      self._error = 1.0 - tf.reduce_sum(errors) / batch_size
    
      if not training:
        self._train_op = tf.no_op()
        return

      self._lr = tf.Variable(0.0, trainable=False)
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),max_grad_norm)
      optimizer = tf.train.RMSPropOptimizer(self._lr)
      self._train_op = optimizer.apply_gradients(zip(grads, tvars))
      
  def step(self, sess, batch):
    """
    Take one step through the data batch.
    Args:
      sess: current tf session the model is being run in
      batch: batch of data of type Batch
    Returns:
      cost: cross entropy cost function for the next batch in batches
      error: binary classifcation error rate for the next batch in batches
      state: the final states of model after a step through the batch
      evals: number of data points evaluated in batch
      predictions: the model predictions for each data point in batch
    """

    feed_dict = dict()

    feed_dict[self._inputs]  = batch.inputs[0]
    feed_dict[self._targets] = batch.targets[0]

    cost, error, predictions, _ = sess.run([self._cost,
                                            self._error,
                                            self._predictions,
                                            self._train_op],
                                             feed_dict)

    #print(predictions)
    #print(error)
    
    return cost, error, self._batch_size, predictions

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self._lr, lr_value))

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
  def lr(self):
    return self._lr
