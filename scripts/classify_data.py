#!/bin/sh
''''exec python3 -u -- "$0" ${1+"$@"} # '''
# The above forces flushed pipes

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
  $ classifiy_data.py --config=train.conf --test_datafile=test.dat > output.dat
  $ paste -d ' ' test.dat output.dat > input_and_output.dat
  """
  configs.DEFINE_string('test_datafile',None,'file with test data')
  configs.DEFINE_string('time_field','date','fields used for dates/time')
  configs.DEFINE_string('print_start','190001','only print data on or after')
  configs.DEFINE_string('print_end','999912','only print data on or before')
  configs.DEFINE_integer('num_batches',None,'num_batches overrride')

  config = configs.get_configs()

  if config.test_datafile is None:
     config.test_datafile = config.datafile

  batch_size = 1
  data_path = model_utils.get_data_path(config.data_dir,config.test_datafile)

  # print("Loading data %s"%data_path)

  dataset = BatchGenerator(data_path, config,
                             batch_size=batch_size,
                             num_unrollings=config.num_unrollings)

  num_data_points = dataset.num_batches
  if config.num_batches is not None:
     num_data_points = config.num_batches

  #print("num_batches = ", num_data_points)

  tf_config = tf.ConfigProto( allow_soft_placement=True,
                                log_device_placement=False )

  with tf.Graph().as_default(), tf.Session(config=tf_config) as session:

    #print("Loading model.")

    model = model_utils.get_trained_model(session, config, verbose=False)

    for i in range(num_data_points):

      batch = dataset.next_batch()
      preds = model.step(session, batch)
      seq_len = get_seq_length(batch)
      key, date = get_key_and_date(batch, seq_len-1)

      if (date < config.print_start or date > config.print_end):
        continue

      score  = get_score(config, preds, seq_len-1)
      target = get_target(config, batch, seq_len-1)

      print("%s %s %.6f %.6f %d" % (key, date, score, target, seq_len))

def get_seq_length(batch):
  return batch.seq_lengths[0]

def get_score(config,preds,idx):
  if config.nn_type != 'rnn':
    idx = 0
  assert(idx < len(preds))
  scorevec = np.linspace(0,1,config.num_outputs)
  score = np.inner(preds[idx],scorevec)
  return score

def get_target(config,batch,idx):
  # k = batch.seq_lengths[0]-1
  scorevec = np.linspace(0,1,config.num_outputs)
  sidx = np.argmax(batch.targets[idx][0])  
  assert(idx < len(batch.targets))
  return scorevec[sidx]

def get_key_and_date(batch,idx):
  # k = batch.seq_lengths[0]-1
  assert(idx < len(batch.attribs))
  key = batch.attribs[idx][0][0]
  date = str(batch.attribs[idx][0][1])
  return key, date

if __name__ == "__main__":
  tf.app.run()
