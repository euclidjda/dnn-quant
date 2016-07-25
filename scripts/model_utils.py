from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import sys

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile
from deep_rnn_model import DeepRnnModel

def get_training_models(session, config, verbose=True):
    mtrain, mvalid, mdeploy = _get_all_models(session,config,verbose)
    return mtrain, mvalid

def get_trained_model(session, config, verbose=False):
    mtrain, mvalid, mdeploy = _get_all_models(session,config,verbose)
    return mdeploy
  
def adjust_learning_rate(session, config, model, learning_rate, batch_perfs):
  if len(batch_perfs) > 5:
    mean = sum(batch_perfs[-5:-1])/4.0
    curr = batch_perfs[-1]
    # If performance has dropped by less than 1%, decay learning_rate
    if ((learning_rate >= 0.001) and (mean > 0.0)
            and (mean >= curr) and (curr/mean >= 0.99)):
        learning_rate = learning_rate * config.lr_decay
  model.assign_lr(session, learning_rate)
  return learning_rate


def _get_all_models(session, config, verbose=False):

    mtrain, mvalid, mdeploy = _create_all_models(session,config,verbose)
    
    ckpt = tf.train.get_checkpoint_state(config.model_dir)
    if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
      if verbose:
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      tf.train.Saver(max_to_keep=200).restore(session,
                                                ckpt.model_checkpoint_path)
    else:
      if verbose:
        print("Created model with fresh parameters.")
      session.run(tf.initialize_all_variables())

    return mtrain, mvalid, mdeploy


def _create_all_models(session,config,verbose=False):
    initer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

    if verbose is True:
      print("Model has the following geometry:")
      print("  num_unroll = %d"% config.num_unrollings)
      print("  batch_size = %d"% config.batch_size)
      print("  stps/epoch = %d"% (config.batch_size*config.num_unrollings))
      print("  num_inputs = %d"% config.num_inputs)
      print("  num_hidden = %d"% config.num_hidden)
      print("  num_layers = %d"% config.num_layers)
      
    with tf.variable_scope("model", reuse=None, initializer=initer), \
         tf.device(config.default_gpu):
      mtrain = DeepRnnModel(training=True, config=config)
    with tf.variable_scope("model", reuse=True, initializer=initer), \
         tf.device(config.default_gpu):
      mvalid = DeepRnnModel(training=False, config=config)

    # TODO: This is bad. We should copy config in case it is used further
    # downstream. Although I got error when trying to copy via copy.copy(config)
    config.num_unrollings = 1
    config.batch_size = 1
    with tf.variable_scope("model", reuse=True, initializer=initer), \
         tf.device(config.default_gpu):
      mdeploy = DeepRnnModel(training=False, config=config)
      
    return mtrain, mvalid, mdeploy
    
