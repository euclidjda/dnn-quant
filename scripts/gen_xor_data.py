#! /usr/bin/env python3

# Copyright 2016 Euclidean Technologies Management LLC All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random

num_points = int(sys.argv[1]) if len(sys.argv)>1 else 100

if (len(sys.argv)>2):
    random.seed(sys.argv[2])

print("id target x1 x2") # print header
id = 0

for i in range(num_points):
    if random.random() < 0.1:
        id += 10
    x1 = random.uniform(-1.0,+1.0)
    x2 = random.uniform(-1.0,+1.0)
    y  = +1.0 if x1*x2 >= 0 else -1.0
    print("%d %+.2f %.6f %.6f"%(id,y,x1,x2))
    
    
    
