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
               that contains the unique id (entity id) for the sequences. 
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
        self._min_seq_length = config.min_seq_length
        self._num_classes = _NUM_CLASSES
        self._num_inputs = config.num_inputs
        self._num_unrollings = num_unrollings
        self._batch_size = batch_size

        self._rnn_loss_weight = None
        if hasattr(config,'rnn_loss_weight'):
            self._rnn_loss_weight = config.rnn_loss_weight
        
        self._config = config # save this around for train_batches() method
        
        if data is None:
            if not os.path.isfile(filename):
                raise RuntimeError("The data file %s does not exists" % filename)
            data = pd.read_csv(filename,sep=' ', dtype={ self._key_name : str } )
            if config.end_date is not None:
                data = data.drop(data[data['date'] > config.end_date].index)

        self._feature_start_idx = list(data.columns.values).index(target_name)+1
        self._key_idx = list(data.columns.values).index(key_name)
        self._target_idx = list(data.columns.values).index(target_name)
        self._date_idx = list(data.columns.values).index('date')
        self._feature_names = list(data.columns.values)[self._target_idx+1:]
        assert(self._feature_start_idx>=0)
        # This assert ensures that no x features are the yval
        assert(list(data.columns.values).index(target_name)
                   < self._feature_start_idx)
        self._data = data
        self._data_len = len(data)
        self._validation_set = dict()
        

        if validation_size is not None:
            if config.seed is not None:
                print("setting random seed to "+str(config.seed))
                random.seed( config.seed )
            # get number of keys
            keys = list(set(data[key_name]))
            keys.sort()
            sample_size = int( config.validation_size * len(keys) )
            sample = random.sample(keys, sample_size)
            self._validation_set = dict(zip(sample,[1]*sample_size))
            print("Num training entities: %d"%(len(keys)-sample_size))
            print("Num validation entities: %d"%sample_size)
            #print("\n".join(sample))
            #exit()

        # Setup indexes into the sequences
        min_seq_length = config.min_seq_length
        self._start_idx = list()
        self._end_idx = list()
        last_key = ""
        cur_length = 1
        for i in range(self._data_len):
            key = data.iat[i,self._key_idx]
            if (key != last_key):
                cur_length = 1
            if (cur_length >= min_seq_length):
                self._end_idx.append(i)
                seq_length = min(cur_length,num_unrollings)
                self._start_idx.append(i-seq_length+1)
                # print("%d %d %d"%(seq_length,i,i-seq_length+1))
            cur_length += 1
            last_key = key

        # Create a cursor of equally spaced indicies into the dataset. Each index
        # in the cursor points to one sequence in a batch and is used to keep
        # track of where we are in the dataset.
        batch_size = self._batch_size
        num_batches = len(self._start_idx) // batch_size
        self._cursor = [ offset * num_batches for offset in range(batch_size) ]
        self._init_cursor = self._cursor[:]
        self._num_batches = num_batches

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
        features_idx = self._feature_start_idx
        num_inputs = self._num_inputs
        key_idx = self._key_idx
        target_idx = self._target_idx
        date_idx = self._date_idx
        for b in range(self._batch_size):
            cursor = self._cursor[b]
            start_idx = self._start_idx[cursor]
            end_idx = self._end_idx[cursor]
            seq_lengths[b] = end_idx-start_idx+1
            assert(seq_lengths[b]>0)
            idx = start_idx + step
            ##### TODO: MOVE THIS OUT OF _next_step()
            ########
            if (idx > end_idx):
                x[b,:] = 0.0
                y[b,:] = 0.0
                train_wghts[b] = 0.0
                valid_wghts[b] = 0.0
                attr.append(None)
            else:
                x[b,:] = data.iloc[idx,features_idx:features_idx+num_inputs].as_matrix()
                val = data.iat[idx,target_idx] # val = +1 or -1
                y[b,0] = abs(val - 1) / 2 # +1 -> 0 and -1 -> 1
                y[b,1] = abs(val + 1) / 2 # -1 -> 0 and +1 -> 1
                date = data.iat[idx,date_idx]
                key = data.iat[idx,key_idx]
                attr.append((key,date))
                weight = 1.0
                if self._rnn_loss_weight is not None:
                    len_minus_one = seq_lengths[b]-1
                    assert(len_minus_one > 0)
                    if (idx == end_idx):
                        weight = self._rnn_loss_weight
                    else:
                        weight = (1.0 - self._rnn_loss_weight) / len_minus_one
                if key not in self._validation_set:
                    train_wghts[b] = weight
                    valid_wghts[b] = 0.0
                else:
                    train_wghts[b] = 0.0
                    valid_wghts[b] = weight
        return x, y, train_wghts, valid_wghts, attr

    def next_batch(self):
        """Generate the next batch of sequences from the data.
        Returns:
          A batch of type Batch (see class def below)
        """
        seq_lengths = np.full(self._batch_size, self._num_unrollings, dtype=int)
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

        #############################################################################
        #   Set cursor for next batch
        #############################################################################
        batch_size = self._batch_size
        if self._randomly_sample is True:
            self._cursor = random.sample(range(len(self._start_idx)),batch_size)
        else:
            num_idxs = len(self._start_idx)
            self._cursor = [ (self._cursor[b]+1)%num_idxs for b in range(batch_size) ]

        return Batch(x_batch, y_batch, seq_lengths,
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

    @property
    def num_unrollings(self):
        return self._num_unrollings
    
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
      train_wghts: Weights that specify an example is in the training data
        and how much they should contribute to the training loss function
      valid_wghts: Weights that specify an example is in the validation data
        set and how much they should contribute to the validation loss
      attribs: Currently this holds a key,date tuple for each data point in
        in the batch
    """

    def __init__(self,inputs,targets,seq_lengths,
                     train_wghts,valid_wghts, attribs):
        self._inputs = inputs
        self._targets = targets
        self._seq_lengths = seq_lengths
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
    def train_wghts(self):
        return self._train_wghts

    @property
    def valid_wghts(self):
        return self._valid_wghts

    @property
    def attribs(self):
        return self._attribs

