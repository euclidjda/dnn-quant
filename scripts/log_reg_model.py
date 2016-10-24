
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
from sklearn.linear_model import LogisticRegression
import pickle

import model_utils

# from tensorflow.python.ops import array_ops

_NUM_OUTPUTS = 2

class LogRegModel(object):
  """
  A Deep MLP Model that supports a binary (two class) output with an
  arbitrary number of fixed width hidden layers.
  """
  def __init__(self, load_from=None):
    with open(load_from, "rb") as f:
        self.clf = pickle.load(f)


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
     X, Y, dates = model_utils.batch_to_tabular(batch)
     Yhat = self.clf.predict_proba(X)
     predictions = np.vstack([Yhat,(1-Yhat)]).T

     return predictions


