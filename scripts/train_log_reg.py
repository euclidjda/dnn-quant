

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
import tensorflow as tf
from sklearn.linear_model import LogisticRegression

import model_utils
from model_utils import get_tabular_data
import configs

from tensorflow.python.platform import gfile
from batch_generator import BatchGenerator


"""
Entry point and main loop for train_net.py. Uses command line arguments to get
model and training specification (see config.py).
"""
configs.DEFINE_string("train_datafile", '',"Training file")
configs.DEFINE_float("lr_decay",0.9, "Learning rate decay")
configs.DEFINE_float("initial_learning_rate",1.0,"Initial learning rate")
configs.DEFINE_float("validation_size",0.0,"Size of validation set as %")
configs.DEFINE_integer("passes",1,"Passes through day per epoch")
configs.DEFINE_integer("max_epoch",0,"Stop after max_epochs")
configs.DEFINE_integer("early_stop",None,"Early stop parameter")
configs.DEFINE_integer("seed",None,"Seed for deterministic training")

config = configs.get_configs()

train_path = model_utils.get_data_path(config.data_dir,config.train_datafile)

print("Loading training data ...")

rand_samp = True if config.use_fixed_k is True else False

data_bg = BatchGenerator(train_path, config,
      config.batch_size, config.num_unrollings,
      validation_size=config.validation_size,
      randomly_sample=False)

train_bg = data_bg.train_batches()
valid_bg = data_bg.valid_batches()

print("Grabbing tabular data from batch generator")
X_train, Y_train = get_tabular_data(train_bg)
X_valid, Y_valid = get_tabular_data(valid_bg)

print("Data processing complete")
############################################################################
#   Perform logistic regression
############################################################################
clf = LogisticRegression(C=1)
print("Training logistic regression classifer")
clf.fit(X_train, Y_train)

training_accuracy = np.mean(clf.predict(X_train) == Y_train)
print("training accuracy: ", training_accuracy)

validation_accuracy = np.mean(clf.predict(X_valid) == Y_valid)
print("validation accuracy: ", validation_accuracy)



