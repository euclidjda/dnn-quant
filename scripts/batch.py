# Copyright 2016 Euclidean Technologies Management LLC  All Rights Reserved.
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

class Batch(object):
    """
    Attributes:

      inputs: The batch's sequences of input values. The number of 
        sequences is equal to batch_size and the physical size of each
        sequence is equal to num_unrollings. The seq_lengths return
        value (see below) might be less than num_unrollings if a sequence 
        ends in less steps than num_unrollings.
      targets: The batch's sequences of target values. The number of 
        sequences is equal to batch_size and the physical size of each
        sequence is equal to num_unrollings.
      seq_lengths: An integer vectors of size batch_size that contains the
        length of each sequence in the batch. The maximum length is
        num_unrollings.
      reset_flags: A binary vector of size batch_size. A value of 0 in
        position n indicates that the data in sequence n of the batch is a
        new entity since the last batch and that the RNN's state should be
        reset.
    """
    
    def __init__(self,inputs,targets,seq_lengths,reset_flags):
        self._inputs = inputs
        self._targets = targets
        self._seq_lengths = seq_lengths
        self._reset_flags = reset_flags

    @property
    def inputs(self):
        return self._inputs

    @property
    def targets(self):
        return self._targets

    @property
    def seq_lengths(self):
        return self._seq_lengths

    @property
    def reset_flags(self):
        return self._reset_flags

    
    
    
        
