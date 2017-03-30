#!/bin/sh
''''exec python3 -u -- "$0" ${1+"$@"} # '''

# #! /usr/bin/env python3
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
import configs

from batch_generator import BatchGenerator

def get_data_path(data_dir, filename):
    """
    Construct the data path for the experiement. If DNN_QUANT_ROOT is
    defined in the environment, then the data path is relative to it.

    Args:
      data_dir: the directory name where experimental data is held
      filename: the data file name
    Returns:
      If DNN_QUANT_ROOT is defined, the fully qualified data path is returned
      Otherwise a path relative to the working directory is returned
    """
    path = os.path.join( data_dir, filename ) 
    # path = data_dir + '/' + filename
    if data_dir != '.' and 'DNN_QUANT_ROOT' in os.environ:
        # path = os.environ['DNN_QUANT_ROOT'] + '/' + path
        path = os.path.join(os.environ['DNN_QUANT_ROOT'], path)
    return path

configs.DEFINE_string("train_datafile", None,"Training file")
configs.DEFINE_float("validation_size",0.0,"Size of validation set as %")
configs.DEFINE_integer("seed",None,"Seed for deterministic training")
configs.DEFINE_float("rnn_loss_weight",None,"How much moret to weight kth example")
config = configs.get_configs()

if config.train_datafile is None:
    config.train_datafile = config.datafile

train_path = get_data_path(config.data_dir,config.train_datafile)

print("Loading batched data ...")

batches = BatchGenerator(train_path, config,
                         config.batch_size,config.num_unrollings,
                         validation_size=config.validation_size,
                         randomly_sample=True)


for i in range(10):
    b = batches.next_batch()
    print("-----------------------------------------------------")
    print("----Atributes: ")
    print(b.attribs)
    print("----Sequence Lengths: ")
    print(b.seq_lengths)
    print("----Train Weights: ")
    print(b.train_mask)
    print("----Valid Weights: ")
    print(b.valid_mask)
    print("----Targets: ")
    print(b.targets)

