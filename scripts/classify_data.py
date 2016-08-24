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
  point in --test_datafile and the model output is sent to --output. The unix
  command 'paste' can be used to stich the input file and output together.
  e.g.,
  $ classifiy_data.py --config=train.conf --test_datafile=test.dat -output=output.dat
  $ paste -d ' ' test.dat output.dat > input_and_output.dat
  """

  configs.DEFINE_string('test_datafile','test.dat','file with test data')
  configs.DEFINE_string('output','preds.dat','file for predictions')
  configs.DEFINE_string('time_name','','fields used for dates/time')

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

    stats = dict()
    key   = 'ALL'

    with open(config.output, "w") as outfile:

    # print the headers so it is easy to "paste" data file and output file
    # together to create a final, single result file
      outfile.write('p0 p1\n')

      for i in range(num_data_points):

        batch = dataset.next_batch()
        cost, error, evals, preds = model.step(session, batch)
        prob = get_pos_prob( preds )

        outfile.write("%.4f %.4f\n" % (1.0 - prob, prob) )

        pred   = +1.0 if prob >= 0.5 else 0.0
        target = get_target(batch)

        if len(config.time_name):
          key = get_time_label(batch, config.time_name)

        tp = 1.0 if (pred==1 and target==1) else 0
        tn = 1.0 if (pred==0 and target==0) else 0
        fp = 1.0 if (pred==1 and target==0) else 0
        fn = 1.0 if (pred==0 and target==1) else 0

        data = { 'cost'  : cost  , 
                 'error' : error ,
                 'tpos'  : tp    ,
                 'tneg'  : tn    ,
                 'fpos'  : fp    ,
                 'fneg'  : fn    }

        if key not in stats:
          stats[key] = list()

        stats[key].append(data)

    print_summary_stats(stats)


def get_pos_prob(preds):
  return preds[0][1]

def get_target(batch):
  return batch.targets[0][0][1]

def get_time_label(batch, time_label):
  # TODO: Make the interface attribs more generic
  return batch.attribs[0][0]

def print_summary_stats(stats):

  keys = [key for key in stats]
  keys.sort()

  for key in keys:

    cost  = 0
    error = 0
    tpos  = 0
    tneg  = 0
    fpos  = 0
    fneg  = 0

    for d in stats[key]:
      cost  += d['cost']
      error += d['error']
      tpos  += d['tpos']
      tneg  += d['tneg']
      fpos  += d['fpos']
      fneg  += d['fneg']

    n = len(stats[key])
    assert(n > 0)

    cost  /= n
    error /= n

    precision = tpos / (tpos+fpos)
    recall    = tpos / (tpos+fneg)

    print("%s loss=%.4f error=%.4f prec=%.4f recall=%.4f" % 
          (key,cost,error,precision,recall))

if __name__ == "__main__":
  tf.app.run()
