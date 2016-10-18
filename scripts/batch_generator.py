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
import random

_NUM_CLASSES = 2

class BatchGenerator(object):
    """
    BatchGenerator object takes a data file are returns an object with
    a next_batch() function. The next_batch() function yields a batch of data 
    sequences from the datafile whose shape is specified by batch_size 
    and num_unrollings.
    """
    def __init__(self,filename,config,
                     batch_size,num_unrollings,
                     validation_size=None,
                     randomly_sample=False,
                     data=None):
        """
        Init a BatchGenerator
        Args:
          filename: Name of file containing data. Each row of the file is a 
            record in a data sequence and each column (space seperated) is a 
            field. The columns have headers that are used to identify the field.
          config: The configuration which should include the following
             config.key_name (entity id): The column with the header key_name is the column
               that contains the unique id (entity id) for the sequences. In the 
               context of an RNN, the state should be reset when an entity ID changes
             config.target_name: The column with the header target_name is the name of
               the column containing the target values (-1 or +1) for the record
             config.num_inputs: The input fields are the first num_inputs columns
               following the target_name column. For example the data might look
               like this if key_name=ID, tareget_name=YY, num_inputs =3:
               ID YY X1 X2 X3
               aa +1 .3 .1 .9
               aa -1 .2 .2 .8
               bb +1 .6 .5 .0
               bb -1 .5 .4 .1
               bb -1 .5 .5 .0
            config.end_date: end_date of training period (inclusive)
          batch_size: The size of a batch (if not defined in config)
          num_unrollings: The time window for each batch. The number of data
            points in a batch is batch_size * num_unrollings
          validation_size: size of validation set 0.10 = 10%
        Raises:
          The file specified by filename does not exist
        """
        key_name = config.key_field
        target_name = config.target_field
        self._key_name = key_name
        self._target_name = target_name
        self._randomly_sample = randomly_sample
        self._use_fixed_k = config.use_fixed_k
        self._num_classes = _NUM_CLASSES
        self._num_inputs = config.num_inputs
        self._num_unrollings = num_unrollings
        self._batch_size = batch_size
        self._config = config # save this around for train_batches() method
        
        if data is None:
            if not os.path.isfile(filename):
                raise RuntimeError("The data file %s does not exists" % filename)
            data = pd.read_csv(filename,sep=' ', dtype={ self._key_name : str } )
            if config.end_date is not None:
                data = data.drop(data[data['date'] > config.end_date].index)

        self._factor_start_idx = list(data.columns.values).index(target_name)+1
        self._key_idx = list(data.columns.values).index(key_name)
        self._target_idx = list(data.columns.values).index(target_name)
        self._date_idx = list(data.columns.values).index('date')
        assert(self._factor_start_idx>=0)
        # This assert ensures that no x factors are the yval
        assert(list(data.columns.values).index(target_name)
                   < self._factor_start_idx)
        self._data = data
        self._data_len = len(data)
        self._validation_set = dict()
        if validation_size is not None:
            if config.seed is not None:
                print("setting random seed to "+str(config.seed))
                random.seed( config.seed )
            # get number of keys
            keys = list(set(data[key_name]))
            sample_size = int( config.validation_size * len(keys) )
            sample = random.sample(keys, sample_size)
            sample.sort()
            self._validation_set = dict(zip(sample,[1]*sample_size))
            print("Num training entities: %d"%(len(keys)-sample_size))
            print("Num validation entities: %d"%sample_size)
            # print(self._validation_set)
        # Create a cursor of equally spaced indicies into the dataset. Each index
        # in the cursor points to one sequence in a batch and is used to keep
        # track of where we are in the dataset.
        batch_size = self._batch_size
        segment = self._data_len // batch_size
        self._init_cursor = [ offset * segment for offset in range(batch_size) ]
        # The following loop ensures that every starting index in the cursor is
        # at the beggining of an entity's time sequences.
        for b in range(batch_size):
            idx = self._init_cursor[b]
            key = data.iat[idx,self._key_idx]    
            while data.iat[idx,self._key_idx] == key:
                # TDO: THIS SHOULD BE FIXED AS IT CAN GO INTO AN INFINITE LOOP
                # IF THERE IS ONLY ONE ENTITY IN DATASET
                idx = (idx + 1) % self._data_len
            if b>0:
                self._init_cursor[b] = idx
        self._cursor = self._init_cursor[:]
        self._num_batches = self._calc_num_batches()

    def _calc_num_batches(self):
        num_batches = 0
        data = self._data
        counts = data.groupby(self._key_name).size()
        unrls = self._num_unrollings
        if self._batch_size == 1 and self._num_unrollings == 1:
            num_batches = self._data_len
        #elif self._randomly_sample is True:
        #    num_batches = self._data_len // (self._batch_size*self._num_unrollings)
        elif self._use_fixed_k:
            for i in range(len(counts)):
                count = counts.iloc[i]
                incr = (count - unrls + 1) if count >= unrls else 1
                num_batches += incr
            num_batches = num_batches // self._batch_size
        else: # self._use_fixed_k is False:
            for i in range(len(counts)):
                count = counts.iloc[i]
                num_batches += count // unrls
                if count % unrls > 0:
                    num_batches += 1
            num_batches = num_batches // self._batch_size
        return num_batches

    def _get_reset_flags(self):
        reset_flags = None
        data = self._data
        reset_flags = np.ones(self._batch_size)
        kidx = self._key_idx
        for b in range(self._batch_size):
            i = self._cursor[b]
            if (i==0 or (data.iat[i,kidx] != data.iat[i-1,kidx])):
                reset_flags[b] = 0.0
        return reset_flags
        
    def _get_next_cursor(self,cur_cursor,saved_cursor):
        assert(len(cur_cursor) == self._batch_size)        
        assert(len(saved_cursor) == self._batch_size)
        next_cursor = cur_cursor[:]
        data = self._data
        kidx = self._key_idx
        for b in range(self._batch_size):
            if (data.iat[cur_cursor[b],kidx] == data.iat[saved_cursor[b],kidx]):
                next_cursor[b] = (saved_cursor[b]+1) % self._data_len
        return next_cursor
    
    def _next_step(self, step, seq_lengths):
        """
        Get next step in current batch.
        """
        x = np.zeros(shape=(self._batch_size, self._num_inputs), dtype=np.float)
        y = np.zeros(shape=(self._batch_size, self._num_classes), dtype=np.float)
        train_wghts = np.zeros(shape=(self._batch_size), dtype=np.float)
        valid_wghts = np.zeros(shape=(self._batch_size), dtype=np.float)
        attr = list()
        data = self._data
        start_idx = self._factor_start_idx
        key_idx = self._key_idx
        target_idx = self._target_idx
        date_idx = self._date_idx
        for b in range(self._batch_size):
            idx = self._cursor[b]
            prev_idx = idx-1 if idx > 0 else self._data_len - 1
            if (step>0 and (data.iat[prev_idx,key_idx] != data.iat[idx,key_idx])):
                x[b,:] = 0.0
                y[b,:] = 0.0
                train_wghts[b] = 0.0
                valid_wghts[b] = 0.0
                if (seq_lengths[b]==self._num_unrollings):
                    seq_lengths[b] = step
                attr.append(None)
            else:
                x[b,:] = data.iloc[idx,start_idx:].as_matrix()
                val = data.iat[idx,target_idx] # val = +1 or -1
                y[b,0] = abs(val - 1) / 2 # +1 -> 0 and -1 -> 1
                y[b,1] = abs(val + 1) / 2 # -1 -> 0 and +1 -> 1
                date = data.iat[idx,date_idx]
                key = data.iat[idx,key_idx]
                attr.append((key,date))
                next_idx = (idx+1) % self._data_len
                if key not in self._validation_set:
                    train_wghts[b] = 1.0
                    valid_wghts[b] = 0.0
                else:
                    train_wghts[b] = 0.0
                    valid_wghts[b] = 1.0
                self._cursor[b] = (self._cursor[b] + 1) % self._data_len
        return x, y, train_wghts, valid_wghts, attr

    def next_batch(self):
        """Generate the next batch of sequences from the data.
        Returns:
          A batch of type Batch (see class def below)
        """
        saved_cursor = self._cursor[:]
        seq_lengths = np.full(self._batch_size, self._num_unrollings, dtype=int)
        reset_flags = self._get_reset_flags()
        x_batch = list()
        y_batch = list()
        train_wghts = list()
        valid_wghts = list()
        attribs = list()
        for i in range(self._num_unrollings):
            x, y, tw, vw, attr = self._next_step(i, seq_lengths)
            x_batch.append(x)
            y_batch.append(y)
            train_wghts.append(tw)
            valid_wghts.append(vw)
            attribs.append(attr)
        if self._randomly_sample is True:
            self._cursor = random.sample(range(self._data_len),self._batch_size)
        elif self._use_fixed_k is True:
            self._cursor = self._get_next_cursor(self._cursor,saved_cursor)
        return Batch(x_batch, y_batch, seq_lengths, reset_flags,
                         train_wghts, valid_wghts, attribs )

    def train_batches(self):
        valid_keys = list(self._validation_set.keys())
        indexes = self._data[self._key_name].isin(valid_keys)
        train_data = self._data[~indexes]
        return BatchGenerator("",self._config,self._batch_size,
                                  self._num_unrollings,
                                  validation_size=None,
                                  randomly_sample=self._randomly_sample,
                                  data=train_data)

    def valid_batches(self):
        valid_keys = list(self._validation_set.keys())
        indexes = self._data[self._key_name].isin(valid_keys)
        valid_data = self._data[indexes]
        return BatchGenerator("",self._config,self._batch_size,
                                  self._num_unrollings,
                                  validation_size=None,
                                  randomly_sample=self._randomly_sample,
                                  data=valid_data)
        
    def num_data_points(self):
        return self._data_len

    def rewind(self):
        self._cursor = self._init_cursor[:]

    @property
    def num_batches(self):
        return self._num_batches
        
class Batch(object):
    """
    A batch object is a container for a subset of data to be processed
    by a model. It has two dimensions: batch_size and num_unrollings.
    batch_size is the number of simulaneous time sequences to process by 
    the model in a batch. num_unrollings is the maximum length
    of each time sequence to process in a batch.
    Since num_unrollings is only relevant to RNN's, num_unrollings should
    be =1 for the MLP models.

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
      train_wghts: Weights that specify an example is in the training data
        and how much they should contribute to the training loss function
      valid_wghts: Weights that specify an example is in the validation data
        set and how much they should contribute to the validation loss
      attribs: Currently this holds a key,date tuple for each data point in
        in the batch
    """
    
    def __init__(self,inputs,targets,seq_lengths,reset_flags,
                     train_wghts,valid_wghts,attribs):
        self._inputs = inputs
        self._targets = targets
        self._seq_lengths = seq_lengths
        self._reset_flags = reset_flags
        self._train_wghts = train_wghts
        self._valid_wghts = valid_wghts
        self._attribs = attribs

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

    @property
    def train_wghts(self):
        return self._train_wghts

    @property
    def valid_wghts(self):
        return self._valid_wghts
    
    @property
    def attribs(self):
        return self._attribs

