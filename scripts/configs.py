# Copyright 2016 Euclidean Technologies Inc. All Rights Reserved.
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

"""Implementation of the configs interface."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

_global_parser = argparse.ArgumentParser()

class _ConfigValues(object):
  """
  Command line argument helper class.
  """
  
  def __init__(self):
    """Global container and accessor for configs and their values."""
    self.__dict__['__configs'] = {}
    self.__dict__['__parsed'] = False

  def _parse_configs(self):
    result, _ = _global_parser.parse_known_args()
    for config_name, val in vars(result).items():
      self.__dict__['__configs'][config_name] = val
    self.__dict__['__parsed'] = True

  def __getattr__(self, name):
    """Retrieves the 'value' attribute of the config --name."""
    if not self.__dict__['__parsed']:
      self._parse_configs()
    if name not in self.__dict__['__configs']:
      raise AttributeError(name)
    return self.__dict__['__configs'][name]

  def __setattr__(self, name, value):
    """Sets the 'value' attribute of the config --name."""
    if not self.__dict__['__parsed']:
      self._parse_configs()
    self.__dict__['__configs'][name] = value


class _LoadFromFile (argparse.Action):
    """Helper that supports the reading of config from a file"""
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            parser.parse_known_args(f.read().split(), namespace)

    
def _define_helper(config_name, default_value, docstring, configtype):
  """Registers 'config_name' with 'default_value' and 'docstring'."""
  _global_parser.add_argument("--" + config_name,
                              default=default_value,
                              help=docstring,
                              type=configtype)

def DEFINE_string(config_name, default_value, docstring):
  """Defines a config of type 'string'.

  Args:
    config_name: The name of the config as a string.
    default_value: The default value the config should take as a string.
    docstring: A helpful message explaining the use of the config.
  """
  _define_helper(config_name, default_value, docstring, str)


def DEFINE_integer(config_name, default_value, docstring):
  """Defines a config of type 'int'.

  Args:
    config_name: The name of the config as a string.
    default_value: The default value the config should take as an int.
    docstring: A helpful message explaining the use of the config.
  """
  _define_helper(config_name, default_value, docstring, int)


def DEFINE_boolean(config_name, default_value, docstring):
  """Defines a config of type 'boolean'.

  Args:
    config_name: The name of the config as a string.
    default_value: The default value the config should take as a boolean.
    docstring: A helpful message explaining the use of the config.
  """
  # Register a custom function for 'bool' so --config=True works.
  def str2bool(v):
    return v.lower() in ('true', 't', '1')
  _global_parser.add_argument('--' + config_name,
                              nargs='?',
                              const=True,
                              help=docstring,
                              default=default_value,
                              type=str2bool)
  _global_parser.add_argument('--no' + config_name,
                              action='store_false',
                              dest=config_name)


# The internal google library defines the following alias, so we match
# the API for consistency.
DEFINE_bool = DEFINE_boolean  # pylint: disable=invalid-name

def DEFINE_float(config_name, default_value, docstring):
  """Defines a config of type 'float'.

  Args:
    config_name: The name of the config as a string.
    default_value: The default value the config should take as a float.
    docstring: A helpful message explaining the use of the config.
  """
  _define_helper(config_name, default_value, docstring, float)

def get_configs():
    """Defines all configuration params passable to command line.
    """
    DEFINE_string("default_gpu",'',"The default GPU to use e.g., /gpu:0")
    DEFINE_string("nn_type",'rnn',"Net Type: mlp or rnn")
    DEFINE_string("key_field", '',"Key column name header in datafile")
    DEFINE_string("target_field", '',"Target column name header in datafile")
    DEFINE_string("data_dir",'',"The data directory")
    DEFINE_string("model_dir", '',"Model directory")
    DEFINE_integer("num_unrollings",1,"Number of unrolling steps")
    DEFINE_integer("batch_size",10,"Size of each batch")
    DEFINE_integer("num_layers",1, "Numer of RNN layers")
    DEFINE_integer("num_inputs",10,"Number of inputs")
    DEFINE_integer("num_hidden",10,"Number of hidden layer units")
    DEFINE_float("init_scale",0.1, "Initial scale for weights")
    DEFINE_float("max_grad_norm",10.0,"Gradient clipping")
    DEFINE_integer("end_date",210001,"Last date to train on as YYYYMM")
    DEFINE_float("keep_prob",1.0,"Keep probability for dropout")
    DEFINE_boolean("use_fixed_k",True,"Sequences fixed at num_unrolling")

    _global_parser.add_argument('--config', type=open,
                                    action=_LoadFromFile,
                                    help="File containing configuration")            

    return _ConfigValues()


