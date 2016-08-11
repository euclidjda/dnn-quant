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

    
    
    
        
