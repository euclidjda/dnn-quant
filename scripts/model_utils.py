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

def stop_training(perfs,num):
    """
    Early stop algorithm

    Args:
      perfs: History of validation performance on each iteration
      num: Number of iterations of validation performance to
        "lookback" to see if performance is improving anymore
    """
    if num is not None:
        far = num+1
        near = min(int(num/2),5)
        if len(perfs) >= far:
            mean1 = np.mean(perfs[-far:-2])
            mean2 = min(np.mean(perfs[-near:-1]),perfs[-1])
            if mean1-mean2 < 0.0:
                return True
    return False
        

def adjust_learning_rate(session, model, learning_rate,
                           lr_decay, batch_perfs, num=5):
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
  num += 1
  if len(batch_perfs) >= num:
    mean = np.mean(batch_perfs[-num:-2])
    curr = batch_perfs[-1]
    # If performance has dropped by less than 1%, decay learning_rate
    if ((learning_rate >= 0.0001) and (mean > 0.0)
            and (mean >= curr) and (curr/mean >= 0.99)):
        learning_rate = learning_rate * lr_decay
  model.assign_lr(session, learning_rate)
  return learning_rate

def get_training_model(session, config, verbose=True):
    mtrain, mdeploy = get_all_models(session, config, verbose)
    return mtrain

def get_trained_model(session, config, verbose=False):
    mtrain, mdeploy = get_all_models(session, config, verbose)
    return mdeploy

def get_all_models(session, config, verbose=False):
    """
    Creates the three graphs for training and testing/deployment.
    If a saved model exists in config.model_dir, read it from file.
    Args:
      session: the tf session
      config: a config that specifies the model geometry and learning params
      verbose: print status output if true
    Returns:
      mtrain:  training model, initialized frm model_dir if exists
      mdeploy: testing/deployment model, initialized frm model_dir if exists
    """
    
    mtrain, mdeploy = _create_all_models(session, config, verbose)
    
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

    return mtrain, mdeploy

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
    elif config.nn_type == 'mlp':
      return _create_all_models_mlp(session,config,verbose)
    else:
      raise RuntimeError("Unknown net_type = %s"%config.nn_type)
  

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

      
    # Training and validation graph  
    with tf.variable_scope("model", reuse=None, initializer=initer), \
      tf.device(config.default_gpu):
        mtrain = DeepRnnModel(num_layers     = config.num_layers,
                              num_inputs     = config.num_inputs,
                              num_hidden     = config.num_hidden,
                              num_unrollings = config.num_unrollings,
                              batch_size     = config.batch_size,
                              max_grad_norm  = config.max_grad_norm)


    # Deployment / testing graph
    with tf.variable_scope("model", reuse=True, initializer=initer), \
      tf.device(config.default_gpu):
        mdeploy = DeepRnnModel(num_layers    = config.num_layers,
                              num_inputs     = config.num_inputs,
                              num_hidden     = config.num_hidden,
                              num_unrollings = 1,
                              batch_size     = 1)
      
    return mtrain, mdeploy


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

    # Deployment / classification graph
    with tf.variable_scope("model", reuse=True, initializer=initer), \
      tf.device(config.default_gpu):
        mdeploy = DeepMlpModel(num_layers    = config.num_layers,
                              num_inputs     = config.num_inputs,
                              num_hidden     = config.num_hidden,
                              batch_size     = 1,
                              training       = False)
      
    return mtrain, mdeploy

