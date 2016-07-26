#! /usr/bin/env python

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

import configs

from tensorflow.python.platform import gfile
from batch_generator import BatchGenerator

def main(_):

  config = configs.CONFIG

  batch_size = 1
  num_unrollings = 1
  
  data = BatchGenerator(config.datafile, model_utils.DATA_FILE_FIELDS,
                          config.num_inputs, config.num_outputs,
                          batch_size, num_unrollings )

  num_data_points = data.num_data_points()
  
  tf_config = tf.ConfigProto( allow_soft_placement=True,
                                log_device_placement=False )

  with tf.Graph().as_default(), tf.Session(config=tf_config) as session:

    model = model_utils.get_trained_model(session, config)
    eval_op = tf.no_op()

    # print the headers so it is easy to "paste" data file and output file
    # together to create a final, single result file
    print('p0 p1')
    
    for i in range(num_data_points):
      xvals, yvals, seq_length, reset_flag = data.next()
      _, _, _, preds = model.step(session, eval_op, xvals, yvals,
                                    seq_length, reset_flag )
      print("%.2f %.2f" % (preds[0][0],preds[0][1]))
      sys.stdout.flush()    

    
if __name__ == "__main__":
  tf.app.run()
