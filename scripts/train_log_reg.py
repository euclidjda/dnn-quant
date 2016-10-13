

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

train_data = BatchGenerator(train_path, config,
      config.batch_size, config.num_unrollings,
      validation_size=config.validation_size,
      randomly_sample=False)

############################################################################
#   Formatting data for logistic regression
############################################################################
X_train = []
Y_train = []

for i in range(train_data.num_batches):
    x = train_data.next_batch()
    inputs = x._inputs
    flat_list = [input[0] for input in inputs]
    X_train.append(np.concatenate(flat_list))
    Y_train.append(x.targets[-1][0,0])


############################################################################
#   Perform logistic regression
############################################################################

clf = LogisticRegression()
clf.fit(X_train, Y_train)

training_accuracy = np.mean(clf.predict(X_train) == Y_train)
print("training accuracy: ", training_accuracy)




