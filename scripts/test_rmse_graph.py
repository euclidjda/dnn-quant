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
    
        for _ in range(UNROLLS):
            g['x'].append( tf.placeholder(tf.float32,shape=[BATCH_SIZE,INPUT_SIZE]) )
            g['y'].append( tf.placeholder(tf.float32,shape=[BATCH_SIZE,INPUT_SIZE]) )
            g['s'].append( tf.placeholder(tf.float32,shape=[BATCH_SIZE]) )

        num_inputs  = INPUT_SIZE * UNROLLS
        num_outputs = OUTPUT_SIZE * UNROLLS
            
        g['w'] = tf.get_variable("softmax_w", [num_inputs,num_outputs])
        g['b'] = tf.get_variable("softmax_b", [num_outputs])

        g['cat_x'] = tf.concat(1, g['x'] )

        g['logits'] = tf.nn.sigmoid( tf.nn.xw_plus_b(g['cat_x'], g['w'], g['b'] ) )

        g['cat_y'] = tf.concat(1, g['y'] )

        g['diff'] = g['logits'] - g['cat_y']

        g['r_s'] = tf.reshape( tf.tile( tf.reshape( tf.transpose( tf.reshape( tf.concat(0, g['s']),
                                                                                 [UNROLLS, BATCH_SIZE] ) ),
                                                    [UNROLLS*BATCH_SIZE,1] ), [1,2] ), [ BATCH_SIZE, num_inputs ] )

        
        g['loss'] =tf.nn.l2_loss(tf.mul(g['diff'],g['r_s']))

        #g['q1'] = tf.tile(tf.reshape( tf.transpose( tf.reshape( tf.concat(0, g['s']), [UNROLLS,BATCH_SIZE] ) ),
         #                         [UNROLLS*BATCH_SIZE,1] ), [1,2] )
        g['q1'] = tf.concat(0, g['s'])
        g['q2'] = g['q1']
        # g['q2'] = tf.tile( g['q1'], [1,2] )
        #g['q2'] = tf.reshape( tf.concat(0, g['s']), [UNROLLS,BATCH_SIZE]  )
        #g['q3'] = tf.transpose( tf.reshape( tf.concat(0, g['s']), [UNROLLS,BATCH_SIZE]  ) )
        #g['q4'] = tf.reshape(tf.transpose( tf.reshape( tf.concat(0, g['s']),
        #                                                    [UNROLLS,BATCH_SIZE]  ) ), [-1] )
        #g['r_s'] = tf.reshape(tf.transpose( tf.reshape( tf.concat(0, g['s']),
        #                                                    [UNROLLS,BATCH_SIZE]  ) ), [-1] )
        
        #g['preds'] = tf.nn.softmax(g['r_logits'])
        
        #g['class_preds'] =  tf.floor( g['preds'] + 0.5 )

        #g['accy'] = tf.mul( g['class_preds'],  g['cat_y'] )

        #g['w_accy'] = tf.mul(g['accy'], tf.reshape(
        #    g['r_s'],
        #    shape=[BATCH_SIZE*UNROLLS,1]) )
        
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
    feed_dict[ G['x'][1] ] = np.array( [ [0.2,0.3], [2,2.5],[0.2,0.3], [2,2.5] ] )
    feed_dict[ G['x'][2] ] = np.array( [ [0.1,0.4], [3.3,3],[0.2,0.3], [2,2.5] ] )

    feed_dict[ G['y'][0] ] = np.array( [ [1,0], [0,1],[1,0], [0,1] ] )
    feed_dict[ G['y'][1] ] = np.array( [ [0,1], [1,0],[1,0], [0,1] ] )
    feed_dict[ G['y'][2] ] = np.array( [ [1,0], [0,1],[1,0], [0,1] ] )

    feed_dict[ G['s'][0] ] = np.array( [ 1, 1, 0, 0 ] )
    feed_dict[ G['s'][1] ] = np.array( [ 2, 2, 0, 0 ] )
    feed_dict[ G['s'][2] ] = np.array( [ 3, 3, 0, 0 ] )
    
    q1, q2, loss, diff, cat_y, logits = session.run( [ G['q1'], G['q2'], G['loss'],G['diff'], G['cat_y'], G['logits'] ],
                                                   feed_dict )
    #w_accy,accy,class_preds,preds,cat_y, r_logits,cat_x,r_s = session.run( [ G['w_accy'],
    #                                                    G['accy'],
    #                                                    G['class_preds'],G['preds'],
    #                                                    G['cat_y'],
    #                                                    G['r_logits'],G['cat_x'],G['r_s']
    #                                                    ],
    #                                              feed_dict )
    print('-'*80)
    print(q1)
    print('-'*80)
    print(q2)
    print('-'*80)
    print(cat_y)
    print('-'*80)
    print(logits)
    print('-'*80)
    print(diff)
    print('-'*80)
    print(loss)
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
