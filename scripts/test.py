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

NUM_INPUTS=2
NUM_HIDDEN=4
NUM_STEPS=3

def create_graph(g):

    g['t_weights'] = tf.reshape(tf.Variable([1.0,1.0,1.0,1.0,2.0,2.0,2.0,2.0,3.0,3.0,3.0,3.0]),[NUM_STEPS,NUM_HIDDEN,1])
    g['f_weights'] = tf.reshape(tf.Variable([10.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0, 1000.0]),[1,NUM_HIDDEN,NUM_INPUTS]) 
    
    # g['t_weights'] = tf.get_variable("t_weights",[NUM_STEPS,NUM_HIDDEN,1])
    # g['f_weights'] = tf.get_variable("f_weights",[1,NUM_HIDDEN,NUM_INPUTS])

    g['mat'] = g['t_weights'] * g['f_weights']
    g['res'] = tf.reshape( g['mat'], [NUM_STEPS*NUM_INPUTS, NUM_HIDDEN] )

def main(_):
  
  tf_config = tf.ConfigProto( allow_soft_placement=True,
                                log_device_placement=False )

  with tf.Graph().as_default(), tf.Session(config=tf_config) as session:

    tf.set_random_seed(100)

    G = dict()
    
    create_graph(G)
    print("graph created.")
    
    session.run( tf.global_variables_initializer() )

    feed_dict = dict()
    
    (t,f,mat,res) = session.run( [ G['t_weights'], G['f_weights'],G['mat'], G['res'] ] , feed_dict )

    # (t,f) = session.run( [ G['t_weights'], G['f_weights'] ] , feed_dict )
    print(t)
    print("--------------------------------------------------------")
    print(f)
    print("--------------------------------------------------------")
    print(mat)
    print("--------------------------------------------------------")
    print(res)
    
if __name__ == "__main__":
  tf.app.run()
