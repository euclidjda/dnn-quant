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
from deep_mlp_model import DeepMlpModel

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
    path = data_dir + '/' + filename
    if 'DNN_QUANT_ROOT' in os.environ:
        path = os.environ['DNN_QUANT_ROOT'] + '/' + path
    return path

def adjust_learning_rate(session, model, learning_rate, lr_decay, batch_perfs):
  """
  Systematically decrease learning rate if current performance is not at
  least 1% better than the moving average performance 

  Args:
    session: the current tf session for training
    model: the deep_rnn_model being trained
    learning_rate: the current learning rate
    lr_decay: the learning rate decay factor
    batch_perfs: list of historical performance
  Returns:
    the updated learning rate being used by the model for training
  """
  if len(batch_perfs) > 5:
    mean = sum(batch_perfs[-5:-1])/4.0
    curr = batch_perfs[-1]
    # If performance has dropped by less than 1%, decay learning_rate
    if ((learning_rate >= 0.0001) and (mean > 0.0)
            and (mean >= curr) and (curr/mean >= 0.99)):
        learning_rate = learning_rate * lr_decay
  model.assign_lr(session, learning_rate)
  return learning_rate

def get_training_models(session, config, verbose=True):
    mtrain, mvalid, mdeploy = get_all_models(session, config, verbose)
    return mtrain, mvalid

def get_trained_model(session, config, verbose=False):
    mtrain, mvalid, mdeploy = get_all_models(session, config, verbose)
    return mdeploy

def get_all_models(session, config, verbose=False):
    """
    Creates the three graphs for training, validation, and testing/deployment.
    If a saved model exists in config.model_dir, read it from file.
    Args:
      session: the tf session
      config: a config that specifies the model geometry and learning params
      verbose: print status output if true
    Returns:
      mtrain:  training model, initialized frm model_dir if exists
      mvalid:  validation model, initialized frm model_dir if exists
      mdeploy: testing/deployment model, initialized frm model_dir if exists
    """
    
    mtrain, mvalid, mdeploy = _create_all_models(session, config, verbose)
    
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
    """
    Creates the three graphs for training, validation, and testing/deployment
    Args:
      session: the tf session
      config: a config that specifies the model geometry and learning params
      verbose: print status output if true
    Returns:
      mtrain:  the training model
      mvalid:  the validation model
      mdeploy: the testing and deployment mode
    """

    if config.nn_type == 'rnn':
      return _create_all_models_rnn(session,config,verbose)
    else:
      return _create_all_models_mlp(session,config,verbose)


def _create_all_models_rnn(session,config,verbose=False):    

    initer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
    
    if verbose is True:
      print("Model has the following geometry:")
      print("  num_unroll  = %d"% config.num_unrollings)
      print("  batch_size  = %d"% config.batch_size)
      print("  evals/batch = %d"% (config.batch_size*config.num_unrollings))
      print("  num_inputs  = %d"% config.num_inputs)
      print("  num_hidden  = %d"% config.num_hidden)
      print("  num_layers  = %d"% config.num_layers)

      
    # Training graph  
    with tf.variable_scope("model", reuse=None, initializer=initer), \
      tf.device(config.default_gpu):
        mtrain = DeepRnnModel(num_layers     = config.num_layers,
                              num_inputs     = config.num_inputs,
                              num_hidden     = config.num_hidden,
                              num_unrollings = config.num_unrollings,
                              batch_size     = config.batch_size,
                              max_grad_norm  = config.max_grad_norm,
                              keep_prob      = config.keep_prob,
                              training       = True)

    # Validation graph (keep prob = 1.0)
    with tf.variable_scope("model", reuse=True, initializer=initer), \
      tf.device(config.default_gpu):
        mvalid = DeepRnnModel(num_layers     = config.num_layers,
                              num_inputs     = config.num_inputs,
                              num_hidden     = config.num_hidden,
                              num_unrollings = config.num_unrollings,
                              batch_size     = config.batch_size,
                              training       = False)

    # Deployment / classification graph
    with tf.variable_scope("model", reuse=True, initializer=initer), \
      tf.device(config.default_gpu):
        mdeploy = DeepRnnModel(num_layers    = config.num_layers,
                              num_inputs     = config.num_inputs,
                              num_hidden     = config.num_hidden,
                              num_unrollings = 1,
                              batch_size     = 1,
                              training       = False)
      
    return mtrain, mvalid, mdeploy


def _create_all_models_mlp(session,config,verbose=False):    

    initer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
    
    if verbose is True:
      print("Model has the following geometry:")
      print("  batch_size  = %d"% config.batch_size)
      print("  num_inputs  = %d"% config.num_inputs)
      print("  num_hidden  = %d"% config.num_hidden)
      print("  num_layers  = %d"% config.num_layers)

    if config.num_unrollings != 1:
        raise RuntimeError("num_unrollings cannot be greater than 1 "+
                               "when training an MLP.")
      
    # Training graph  
    with tf.variable_scope("model", reuse=None, initializer=initer), \
      tf.device(config.default_gpu):
        mtrain = DeepMlpModel(num_layers     = config.num_layers,
                              num_inputs     = config.num_inputs,
                              num_hidden     = config.num_hidden,
                              batch_size     = config.batch_size,
                              max_grad_norm  = config.max_grad_norm,
                              keep_prob      = config.keep_prob,
                              training       = True)

    # Validation graph (keep prob = 1.0)
    with tf.variable_scope("model", reuse=True, initializer=initer), \
      tf.device(config.default_gpu):
        mvalid = DeepMlpModel(num_layers     = config.num_layers,
                              num_inputs     = config.num_inputs,
                              num_hidden     = config.num_hidden,
                              batch_size     = config.batch_size,
                              training       = False)
    
    # Deployment / classification graph
    with tf.variable_scope("model", reuse=True, initializer=initer), \
      tf.device(config.default_gpu):
        mdeploy = DeepMlpModel(num_layers    = config.num_layers,
                              num_inputs     = config.num_inputs,
                              num_hidden     = config.num_hidden,
                              batch_size     = 1,
                              training       = False)
      
    return mtrain, mvalid, mdeploy

