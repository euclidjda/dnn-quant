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

class DeepMlpModel(object):
  """
  A Deep MLP Model that supports a binary (two class) output with an
  arbitrary number of fixed width hidden layers.
  """
  def __init__(self, num_layers, num_inputs, num_hidden,
               num_unrollings, batch_size,
               max_grad_norm=5.0,
               input_dropout=False):
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
        input_dropout: perform dropout on input layer
      """
      self._batch_size = batch_size
      self._num_unrollings = num_unrollings
      num_outputs = _NUM_OUTPUTS

      total_input_size = num_unrollings * num_inputs

      self._seq_lengths = tf.placeholder(tf.int64, shape=[batch_size])
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

      inputs = tf.concat( 1, self._inputs )
      outputs = inputs
      if input_dropout is True:
        outputs = tf.nn.dropout(outputs, self._keep_prob)
      num_prev = total_input_size

      for i in range(num_layers):
        weights = tf.get_variable("hidden_w_%d"%i,[num_prev, num_hidden])
        biases = tf.get_variable("hidden_b_%d"%i,[num_hidden])
        outputs = tf.nn.relu(tf.nn.xw_plus_b(outputs, weights, biases))
        outputs = tf.nn.dropout(outputs, self._keep_prob)
        num_prev = num_hidden

      softmax_w = tf.get_variable("softmax_w", 
                                  [total_input_size+num_prev, num_outputs])

      softmax_b = tf.get_variable("softmax_b", [num_outputs])

      logits = tf.nn.xw_plus_b( tf.concat( 1, [ inputs, outputs]), 
                                softmax_w, softmax_b)

      targets = tf.unpack(tf.reverse_sequence(tf.reshape(
        tf.concat(1, self._targets ),[batch_size,num_unrollings,num_outputs] ),
        self._seq_lengths,1,0),axis=1)[0]

      agg_loss = tf.nn.softmax_cross_entropy_with_logits(logits,targets)

      train_wghts = tf.unpack(tf.reverse_sequence(tf.transpose(
        tf.reshape( tf.concat(0, self._train_wghts ),
        [num_unrollings, batch_size] ) ),
        self._seq_lengths,1,0),axis=1)[0] 

      valid_wghts = tf.unpack(tf.reverse_sequence(tf.transpose(
            tf.reshape( tf.concat(0, self._valid_wghts ),
            [num_unrollings, batch_size] ) ),
            self._seq_lengths,1,0),axis=1)[0] 

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
                                          shape=[batch_size,1]))
      valid_accy = tf.mul(accy,tf.reshape(valid_wghts,
                                          shape=[batch_size,1]))

      self._train_accy = tf.reduce_sum( train_accy )
      self._valid_accy = tf.reduce_sum( valid_accy )

      self._cost  = self._train_cst
      self._accy  = self._train_accy
      self._evals = self._train_evals
      self._batch_cst = self._train_cst / (self._train_evals + 1.0)

      # here is the learning part of the graph
      self._lr = tf.Variable(0.0, trainable=False)
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(self._batch_cst,
                                                         tvars),max_grad_norm)
      # optimizer = tf.train.GradientDescentOptimizer(self.lr)
      optimizer = tf.train.AdagradOptimizer(self.lr)
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

    # assert( train_evals > 0 )
    
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

     feed_dict = self._get_feed_dict(batch)

     predictions = sess.run(self._predictions,feed_dict)

     return predictions

   
  def _get_feed_dict(self,batch,keep_prob=1.0):

    feed_dict = dict()

    feed_dict[self._keep_prob] = keep_prob
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
  def lr(self):
    return self._lr

  @property
  def num_unrollings(self):
    return self._num_unrollings
