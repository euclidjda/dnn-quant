# Copyright 2015 Euclidean Technologies Management LLC  All Rights Reserved.
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

import numpy as np
import pandas as pd

class BatchGenerator(object):

    def __init__(self,filename,key_name,target_name,num_inputs,num_classes,
                     batch_size,num_unrollings):
        data = pd.read_csv(filename,sep=' ')
        self._num_inputs = num_inputs
        self._key_name = key_name = key_name
        self._yval_name = yval_name = target_name
        self._factor_start_idx = list(data.columns.values).index(target_name)+1
        assert(self._factor_start_idx>=0)
        # This assert ensures that no x factors are the yval
        assert(list(data.columns.values).index(yval_name)
                   < self._factor_start_idx)
        self._num_classes = num_classes
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
        self._num_steps = self._calc_num_steps()

    def _proc_headers(self,headers):
        vals = headers.split(',')
        self._key_name = vals[0]
        self._yval_name = vals[2]
        
    def _calc_num_steps(self):
        tmp_cursor = self._cursor[:] # copy cursor
        self.rewind_cursor()
        end_idx = self._init_cursor[1] if self._batch_size > 1 else self._data_size-1
        num_steps = 0
        while (self._cursor[0] < end_idx - self._num_unrollings):
            self.next()
            num_steps += 1
        self._cursor = tmp_cursor[:]
        return num_steps
        
    def _get_reset_flags(self):
        data = self._data
        key_name = self._key_name
        reset_flags = np.ones(self._batch_size)
        for b in range(self._batch_size):
            idx = self._cursor[b]
            if (idx==0 or (data.loc[idx,key_name] != data.loc[idx-1,key_name])):
                reset_flags[b] = 0.0
        return reset_flags
        
    def _next_batch(self,step):
        x_batch = np.zeros(shape=(self._batch_size, self._num_inputs), dtype=np.float)
        y_batch = np.zeros(shape=(self._batch_size, self._num_classes), dtype=np.float)
        data = self._data
        start_idx = self._factor_start_idx
        key_name = self._key_name
        yval_name = self._yval_name
        for b in range(self._batch_size):
            idx = self._cursor[b]
            if (step>0 and idx>0 and (data.loc[idx-1,key_name] != data.loc[idx,key_name])):
                x_batch[b,:] = 0.0
                y_batch[b,:] = 0.0
                if (self._seq_lengths[b]==self._num_unrollings):
                    self._seq_lengths[b] = step
            else:
                x_batch[b,:] = data.iloc[idx,start_idx:].as_matrix()
                val = data.loc[idx,yval_name] # +1, -1
                y_batch[b,0] = abs(val - 1) / 2
                y_batch[b,1] = abs(val + 1) / 2
                self._cursor[b] = (self._cursor[b] + 1) % self._data_size
        return x_batch,y_batch

    def next(self):
        """Generate the next array of batches from the data.
        """
        self._seq_lengths[:] = self._num_unrollings
        reset_flags = self._get_reset_flags()
        # print(self._seq_lengths)
        x_batches = list()
        y_batches = list()
        for step in range(self._num_unrollings):
            x, y = self._next_batch(step)
            x_batches.append(x)
            y_batches.append(y)
        return x_batches, y_batches, self._seq_lengths, reset_flags

    def num_data_points(self):
        return self._data_size

    def is_predata(self,lag):
        assert(self._num_unrollings==1)
        assert(self._batch_size==1)
        assert(lag >= 1)
        lag      = lag - 1
        data     = self._data
        data_size= self._data_size
        key_name = self._key_name
        cur_idx  = self._cursor[0]
        lag_idx  = cur_idx - lag if cur_idx >= lag else data_size-1 
        cur_key  = data.loc[cur_idx,key_name]
        lag_key  = data.loc[lag_idx,key_name]
        if cur_key != lag_key:
            return True
        else:
            return False

    def rewind_cursor(self):
        self._cursor = self._init_cursor[:]

    @property
    def num_steps(self):
        return self._num_steps
        
