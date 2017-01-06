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

SEED = 10000
INIT_SCALE = 0.1
UNROLLS    = 3
BATCH_SIZE = 4
INPUT_SIZE = 2
OUTPUT_SIZE= 2 

def create_graph(g):
    initer = tf.random_uniform_initializer(0.0,INIT_SCALE)

    with tf.variable_scope("graph", reuse=None, initializer=initer):
        g['x'] = list()
        g['y'] = list()
        g['s'] = list()
        g['seq_lengths'] = tf.placeholder(tf.int64,shape=[BATCH_SIZE]);
    
        for _ in range(UNROLLS):
            g['x'].append( tf.placeholder(tf.float32,shape=[BATCH_SIZE,INPUT_SIZE]) )
            g['y'].append( tf.placeholder(tf.float32,shape=[BATCH_SIZE,INPUT_SIZE]) )
            g['s'].append( tf.placeholder(tf.float32,shape=[BATCH_SIZE]) )

        num_inputs  = INPUT_SIZE * UNROLLS
        # num_outputs = OUTPUT_SIZE * UNROLLS
            
        g['w'] = tf.get_variable("softmax_w", [num_inputs,OUTPUT_SIZE])
        g['b'] = tf.get_variable("softmax_b", [OUTPUT_SIZE])

        g['cat_x'] = tf.concat(1, g['x'] )

        g['logits'] = tf.nn.xw_plus_b(g['cat_x'], g['w'], g['b'] )
        
        g['cat_y'] = tf.unpack(tf.reverse_sequence(tf.reshape( tf.concat(1, g['y'] ),
            [BATCH_SIZE,UNROLLS,OUTPUT_SIZE] ),g['seq_lengths'],1,0),axis=1)[0]

        g['loss'] = tf.nn.softmax_cross_entropy_with_logits(g['logits'], g['cat_y'])

        g['r_s'] = tf.unpack(tf.reverse_sequence(tf.transpose(
            tf.reshape( tf.concat(0, g['s'] ), [UNROLLS, BATCH_SIZE] ) ),
            g['seq_lengths'],1,0),axis=1)[0]

        g['train_loss'] = tf.mul( g['loss'], g['r_s'] )
        
        g['preds'] = tf.nn.softmax(g['logits'])
        
        g['class_preds'] =  tf.floor( g['preds'] + 0.5 )

        g['accy'] = tf.mul( g['class_preds'],  g['cat_y'] )

        g['w_accy'] = tf.mul(g['accy'], tf.reshape(g['r_s'],shape=[BATCH_SIZE,1]))
        
def main(_):
  
  tf_config = tf.ConfigProto( allow_soft_placement=True,
                                log_device_placement=False )

  with tf.Graph().as_default(), tf.Session(config=tf_config) as session:

    if SEED is not None:
      tf.set_random_seed(SEED)

    G = dict()
    
    create_graph(G)
    print("graph created.")
    
    session.run( tf.initialize_all_variables() )

    feed_dict = dict()
    feed_dict[ G['x'][0] ] = np.array( [ [0.2,0.3], [2,2.5],[0.2,0.3], [2,2.5] ] )
    feed_dict[ G['x'][1] ] = np.array( [ [0.2,0.3], [2,2.5],[0.2,0.3], [0,0.0] ] )
    feed_dict[ G['x'][2] ] = np.array( [ [0.1,0.4], [3.3,3],[0.0,0.0], [0,0.0] ] )

    feed_dict[ G['y'][0] ] = np.array( [ [1,0], [1,0],[1,0], [0,1] ] )
    feed_dict[ G['y'][1] ] = np.array( [ [0,1], [1,0],[0,1], [0,0] ] )
    feed_dict[ G['y'][2] ] = np.array( [ [0,1], [0,1],[0,0], [0,0] ] )

    feed_dict[ G['seq_lengths'] ] = np.array( [ 3, 3, 2, 1 ] )
    
    feed_dict[ G['s'][0] ] = np.array( [ 1, 1, 0, 0 ] )
    feed_dict[ G['s'][1] ] = np.array( [ 1, 1, 0, 0 ] )
    feed_dict[ G['s'][2] ] = np.array( [ 1, 1, 0, 0 ] )
    
    w_accy, r_s, accy, cat_y, class_preds = session.run( [ G['w_accy'], G['r_s'], G['accy'], G['cat_y'],G['class_preds'] ], feed_dict )
    #w_accy,accy,class_preds,preds,cat_y, r_logits,cat_x,r_s = session.run( [ G['w_accy'],
    #                                                    G['accy'],
    #                                                    G['class_preds'],G['preds'],
    #                                                    G['cat_y'],
    #                                                    G['r_logits'],G['cat_x'],G['r_s']
    #                                                    ],
    #                                              feed_dict )
    print('-'*80)
    print(cat_y)
    print('-'*80)
    print(class_preds)
    print('-'*80)
    print(accy)
    print('-'*80)
    print(r_s)
    print('-'*80)
    print(w_accy)


    #print('-'*80)
    #print(cat_x)
    #print('-'*80)
    #print(r_logits)
    #print('-'*80)
    #print(preds)
    # print(Y)
    #print('-'*80)
    # print(logits)
    # print('-'*80)
    # print(loss)
    
if __name__ == "__main__":
  tf.app.run()
