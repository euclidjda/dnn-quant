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

import os
import numpy as np
import pandas as pd

_NUM_CLASSES = 2

class BatchGenerator(object):
    """
    BatchGenerator object takes a data file are returns an object with
    a next_batch() function. The next_batch() function yields a batch of data 
    sequences from the datafile whose shape is specified by batch_size 
    and num_unrollings.
    """
    def __init__(self,filename,key_name,target_name,num_inputs,
                     batch_size,num_unrollings):
        """
        Init a BatchGenerator
        Args:
          filename: Name of file containing data. Each row of the file is a 
            record in a data sequence and each column (space seperated) is a 
            field. The columns have headers that are used to identify the field.
          key_name: The column with the header key_name is the column
            that contains the unique id for the sequences.
          target_name: The column with the header target_name is the name of
            the column containing the target values (-1 or +1) for the record
          num_inputs: The input fields are the first num_inputs columns
            following the target_name column. For example the data might look
            like this if key_name=ID, tareget_name=YY, num_inputs =3:
            ID YY X1 X2 X3
            aa +1 .3 .1 .9
            aa +1 .2 .2 .8
            bb +1 .6 .5 .0
            bb -1 .5 .4 .1
            bb -1 .5 .5 .0
          batch_size: The size of a batch
          num_unrollings: The time window for each batch. The number of data
            points in a batch is batch_size * num_unrollings
        Raises:
          The file specified by filename does not exist
        """
        if not os.path.exists(filename):
            raise RuntimeError("The data file %s does not exists" % filename)
        data = pd.read_csv(filename,sep=' ')
        self._num_inputs = num_inputs
        self._key_name = key_name = key_name
        self._yval_name = yval_name = target_name
        self._factor_start_idx = list(data.columns.values).index(target_name)+1
        assert(self._factor_start_idx>=0)
        # This assert ensures that no x factors are the yval
        assert(list(data.columns.values).index(yval_name)
                   < self._factor_start_idx)
        self._num_classes = _NUM_CLASSES
        self._num_unrollings = num_unrollings
        self._batch_size = batch_size
        self._data = data
        self._data_size = len(data)
        self._seq_lengths = np.empty( self._batch_size, dtype=int )
        segment = self._data_size // batch_size
        self._init_cursor = [ offset * segment for offset in range(batch_size) ]
        for b in range(batch_size):
            idx = self._init_cursor[b]
            key = data.loc[idx,key_name]    
            while data.loc[idx,key_name] == key:
                # Warning: THIS CAN GO INTO AN INFINITE LOOP
                # IF THERE IS ONLY ONE SECURITY IN DATASET
                idx = (idx + 1) % self._data_size
            if b>0:
                self._init_cursor[b] = idx
        self._cursor = self._init_cursor[:]
        self._num_batches = self._calc_num_batches()

    def _calc_num_batches(self):
        tmp_cursor = self._cursor[:] # copy cursor
        self.rewind()
        end_idx = self._init_cursor[1] if self._batch_size > 1 else self._data_size-1
        num_batches = 0
        while (self._cursor[0] < end_idx - self._num_unrollings):
            self.next_batch()
            num_batches += 1
        self._cursor = tmp_cursor[:]
        return num_batches
        
    def _get_reset_flags(self):
        data = self._data
        key_name = self._key_name
        reset_flags = np.ones(self._batch_size)
        for b in range(self._batch_size):
            idx = self._cursor[b]
            if (idx==0 or (data.loc[idx,key_name] != data.loc[idx-1,key_name])):
                reset_flags[b] = 0.0
        return reset_flags
        
    def _next_step(self,step):
        """
        Get next step in current batch.
        """
        x = np.zeros(shape=(self._batch_size, self._num_inputs), dtype=np.float)
        y = np.zeros(shape=(self._batch_size, self._num_classes), dtype=np.float)
        data = self._data
        start_idx = self._factor_start_idx
        key_name = self._key_name
        yval_name = self._yval_name
        for b in range(self._batch_size):
            idx = self._cursor[b]
            if (step>0 and idx>0 and (data.loc[idx-1,key_name] != data.loc[idx,key_name])):
                x[b,:] = 0.0
                y[b,:] = 0.0
                if (self._seq_lengths[b]==self._num_unrollings):
                    self._seq_lengths[b] = step
            else:
                x[b,:] = data.iloc[idx,start_idx:].as_matrix()
                val = data.loc[idx,yval_name] # +1, -1
                y[b,0] = abs(val - 1) / 2 # +1 -> 0 & -1 -> 1
                y[b,1] = abs(val + 1) / 2 # -1 -> 0 & +1 -> 1
                self._cursor[b] = (self._cursor[b] + 1) % self._data_size
        return x, y

    def next_batch(self):
        """Generate the next batch of sequences from the data.
        Returns:
          A batch of type Batch (see class def below)
        """
        self._seq_lengths[:] = self._num_unrollings
        reset_flags = self._get_reset_flags()
        x_batch = list()
        y_batch = list()
        for step in range(self._num_unrollings):
            x, y = self._next_step(step)
            x_batch.append(x)
            y_batch.append(y)
        return Batch(x_batch,y_batch,self._seq_lengths,reset_flags)

    def num_data_points(self):
        return self._data_size

    def rewind(self):
        self._cursor = self._init_cursor[:]

    @property
    def num_batches(self):
        return self._num_batches
        
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

