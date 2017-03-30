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

class DeepNNModel(object):
  """
  """

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

    (x,y) = sess.run([self._grad_norm,self._grad_norm],feed_dict)
    print("%.2f %.2f"%(x,y))
    # exit()

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
      feed_dict[self._train_mask[i]] = batch.train_mask[i]
      feed_dict[self._valid_mask[i]] = batch.valid_mask[i]
    
    return feed_dict

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self._lr, lr_value))
    return lr_value

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
