# dnn-quant

Tool for building deep / recurrent neural network models for systematic fundamental investing.

Once cloned, the following should be done to setup and test installation:

cd dnn-quant

sudo pip3 install -r requirements.txt

./scripts/setup.py

cd experiments/system-test

train_net.py --config=system-test.conf

classify_data.py --config=system-test.conf > test.out

