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
  $ classifiy_data.py --config=train.conf --test_datafile=test.dat -output=output.dat
  $ paste -d ' ' test.dat output.dat > input_and_output.dat
  """
  configs.DEFINE_string('test_datafile','test.dat','file with test data')
  configs.DEFINE_string('output','preds.dat','file for predictions')
  configs.DEFINE_string('time_field','date','fields used for dates/time')
  configs.DEFINE_integer('print_start',190001,'only print data on or after')
  configs.DEFINE_integer('print_end',210001,'only print data on or before')
  configs.DEFINE_integer('min_test_k',1,'minimum seq length classified')

  config = configs.get_configs()

  batch_size = 1
  num_unrollings = config.num_unrollings
  if (config.use_fixed_k is True) and (config.nn_type == 'rnn'):
    num_unrollings = 1
  
  data_path = model_utils.get_data_path(config.data_dir,config.test_datafile)

  print("Loading data %s"%data_path)

  dataset = BatchGenerator(data_path, config,
                             batch_size=batch_size,
                             num_unrollings=num_unrollings)

  num_data_points = dataset.num_batches
  print("num_batches = ", num_data_points)

  tf_config = tf.ConfigProto( allow_soft_placement=True,
                                log_device_placement=False )

  with tf.Graph().as_default(), tf.Session(config=tf_config) as session:

    print("Loading model.")

    model = model_utils.get_trained_model(session, config)

    stats = dict()
    key   = 0
    stats[key] = list()

    with open(config.output, "w") as outfile:

      for i in range(num_data_points):

        cur_time = time.time()
        batch = dataset.next_batch()
        #print("batch-time: %.8f"%(time.time()-cur_time))
        cur_time = time.time()
        preds = model.step(session, batch)
        #print("step-time: %.8f"%(time.time()-cur_time))
        seq_len = get_seq_length( batch )
        start = 0 if use_entire_seq(batch) is True else seq_len-1
        for i in range(start,seq_len):
          if i+1 < config.min_test_k:
            continue
          prob = get_pos_prob( preds, batch, i )
          key, date = get_key_and_date( batch, i )
          target = get_target(batch, i)
          outfile.write("%s %s "
            "%.4f %.4f %d %d\n" % (key, date, 1.0 - prob, prob, target, i+1) )
          pred   = +1.0 if prob >= 0.5 else 0.0
          error = 0.0 if (pred == target) else 1.0
          tp = 1.0 if (pred==1 and target==1) else 0.0
          tn = 1.0 if (pred==0 and target==0) else 0.0
          fp = 1.0 if (pred==1 and target==0) else 0.0
          fn = 1.0 if (pred==0 and target==1) else 0.0
          # print("pred=%.2f target=%.2f tp=%d tn=%d fp=%d fn=%d"%(pred,target,tp,tn,fp,fn))
          data = { 
	      'error' : error ,
	      'tpos'  : tp    ,
	      'tneg'  : tn    ,
	      'fpos'  : fp    ,
	      'fneg'  : fn    }
          if date not in stats:
            stats[date] = list()
          stats[date].append(data)
          stats[0].append(data)

    print_summary_stats(stats,config.print_start,config.print_end)


def get_seq_length(batch):
  return batch.seq_lengths[0]

def use_entire_seq(batch):
  rf = batch.reset_flags[0]
  return True if rf == 0.0 else False

def get_pos_prob(preds,batch,idx):
  # k = batch.seq_lengths[0]-1
  assert(idx < len(preds))
  return preds[idx][1]

def get_target(batch,idx):
  # k = batch.seq_lengths[0]-1
  assert(idx < len(batch.targets))
  return batch.targets[idx][0][1]

def get_key_and_date(batch,idx):
  # k = batch.seq_lengths[0]-1
  assert(idx < len(batch.attribs))
  return batch.attribs[idx][0][0],batch.attribs[idx][0][1]

def print_summary_stats(stats,print_start,print_end):

  dates = [date for date in stats]
  dates.sort()

  for date in dates:
    if date < print_start or date > print_end:
       continue

    error = 0
    tpos  = 0
    tneg  = 0
    fpos  = 0
    fneg  = 0

    for d in stats[date]:
      error += d['error']
      tpos  += d['tpos']
      tneg  += d['tneg']
      fpos  += d['fpos']
      fneg  += d['fneg']

    n = len(stats[date])
    assert(n > 0)

    error /= n

    precision = "NA"
    if tpos+fpos > 0:
      precision = "%.4f"% (tpos/(tpos+fpos))

    recall = "NA"
    if tpos+fneg > 0:
      recall = "%.4f"%(tpos/(tpos+fneg))

    print("%s cnt=%d error=%.4f prec=%s recall=%s" % 
          (date,n,error,precision,recall))

if __name__ == "__main__":
  tf.app.run()
