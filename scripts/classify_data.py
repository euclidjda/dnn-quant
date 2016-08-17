#! /usr/bin/env python3

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

import model_utils
import configs

from tensorflow.python.platform import gfile
from batch_generator import BatchGenerator

def main(_):
  """
  The model specified command line arg --model_dir is applied to every data
  point in --test_datafile and the model output is set to stdout. The unix
  command 'paste' can be used to stich the input file and output together.
  e.g.,
  $ classifiy_data.py --config=train.conf --test_datafile=test.dat > output.dat
  $ paste -d ' ' test.dat output.dat > input_and_output.dat
  """
  config = configs.get_configs()

  batch_size = 1
  num_unrollings = 1

  data_path = model_utils.get_data_path(config.data_dir,config.test_datafile)
  
  dataset = BatchGenerator(data_path,
                            config.key_name, config.target_name,
                            config.num_inputs,
                            batch_size, num_unrollings )

  num_data_points = dataset.num_data_points()
  
  tf_config = tf.ConfigProto( allow_soft_placement=True,
                                log_device_placement=False )

  with tf.Graph().as_default(), tf.Session(config=tf_config) as session:

    model = model_utils.get_trained_model(session, config)

    # print the headers so it is easy to "paste" data file and output file
    # together to create a final, single result file
    print('p0 p1')
    
    for i in range(num_data_points):
      batch = dataset.next_batch()
      _, _, preds = model.step(session, batch )
      print("%.4f %.4f" % (preds[0][0],preds[0][1]))
      sys.stdout.flush()    

    
if __name__ == "__main__":
  tf.app.run()
