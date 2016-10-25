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
import pandas as pd

import model_utils
import configs

def main(_):
  """
  """
  configs.DEFINE_string('test_datafile',None,'file with test data')
  configs.DEFINE_string('output','preds.dat','file for predictions')
  configs.DEFINE_string('time_field','date','fields used for dates/time')
  configs.DEFINE_string('print_start','190001','only print data on or after')
  configs.DEFINE_string('print_end','210012','only print data on or before')
  configs.DEFINE_string('factor_name',None,'Name of factor if nn_type=factor')
  configs.DEFINE_integer('min_test_k',1,'minimum seq length classified')
  configs.DEFINE_integer('num_batches',None,'num_batches overrride')

  config = configs.get_configs()

  factor_name = config.factor_name
  assert(factor_name is not None)
  
  if config.test_datafile is None:
    config.test_datafile = config.datafile
  batch_size = 1
  num_unrollings = config.num_unrollings
  data_path = model_utils.get_data_path(config.data_dir,config.test_datafile)
  filename=data_path
  
  print("Loading data %s"%data_path)
  if not os.path.isfile(filename):
    raise RuntimeError("The data file %s does not exists" % filename)
  data = pd.read_csv(filename,sep=' ',
		       dtype={ config.key_field : str, 'date' : str } )
  if config.end_date is not None:
    data = data.drop(data[data['date'] > str(config.end_date)].index)

  num_data_points = len(data)

  params = dict()  
     
  print("num data points = ", num_data_points)

  stats = dict()
  key   = 'ALL'
  stats[key] = list()

  with open(config.output, "w") as outfile:

    last_key = ''
    seq_len = 0
    
    for i in range(num_data_points):
      key = get_value( data, config.key_field, i )
      date = get_value( data, 'date', i )
      seq_len = seq_len + 1 if key == last_key else 1
      last_key = key
      if (str(date) < config.print_start or str(date) > config.print_end):
        continue
      if seq_len < config.min_test_k:
        continue
      prob = get_value(data, factor_name, i )
      out = get_value(data, config.target_field, i )
      target = (out+1.0)/2.0
      k = min(seq_len,config.num_unrollings)
      outfile.write("%s %s "
        "%.4f %.4f %d %d\n" % (key, date, 1.0 - prob, prob, target, k) )
      pred   = +1.0 if prob >= 0.5 else 0.0
      error = 0.0 if (pred == target) else 1.0
      tpos = 1.0 if (pred==1 and target==1) else 0.0
      tneg = 1.0 if (pred==0 and target==0) else 0.0
      fpos = 1.0 if (pred==1 and target==0) else 0.0
      fneg = 1.0 if (pred==0 and target==1) else 0.0
      # print("pred=%.2f target=%.2f tp=%d tn=%d fp=%d fn=%d"%(pred,target,tp,tn,fp,fn))
      rec = {
        'error' : error ,
	'tpos'  : tpos  ,
	'tneg'  : tneg  ,
	'fpos'  : fpos  ,
	'fneg'  : fneg  }
      if date not in stats:
        stats[date] = list()
      stats[date].append(rec)
      stats['ALL'].append(rec)

  print_summary_stats(stats)

def get_value(data,name,idx):
  value = data.ix[idx,name]
  return value

def print_summary_stats(stats):

  dates = [date for date in stats]
  dates.sort()

  for date in dates:
    #print(date)
    #print(type(date))
    #print(len(stats[date]))
    #exit()
      
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
