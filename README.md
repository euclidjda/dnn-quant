# dnn-quant

Tool for building deep / recurrent neural network models for systematic fundamental investing.

## Installation and Setup

You will need to have a working installation of tensorflow for your platform.
See https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html

Clone this repository with:

```shell
$ git clone git@github.com:euclidjda/dnn-quant.git
```

To install prerequisites, setup your enviroment, and test the system run the following commands:

```shell
$ cd dnn-quant
$ sudo pip3 install -r requirements.txt
$ ./scripts/setup.py
$ cd exprmnts/system-tests
$ train_net.py --config=system-test.conf
```

### Tested platforms include:
- python 3.4, Ubuntu 14.04.5 LTS, with GPU, tensorflow r0.10 & tensorflow r0.08
- python 3.5, Mac OSX 10.11.5, no GPU, tensorflow r0.10 & tensorflow r0.08


---


## Running the System Test Experiments

```shell
$ cd exprmnts/system-tests
$ train_net.py --config=system-test.conf
$ classify_data.py --config=system-test.conf --test_datafile=mlp-xor-test.dat --output=preds.dat
$ paste -d ' ' $DNN_QUANT_ROOT/datasets/mlp-xor-test.dat preds.dat  > results.dat
$ head results.dat
```

Which should output a space seperated file that looks like:

```shell
id target x1 x2 p0 p1
1 +1.00 -0.446373 -0.715276 0.0000 1.0000
2 +1.00 0.149692 0.896433 0.0000 1.0000
3 +1.00 -0.803404 -0.377976 0.0000 1.0000
4 -1.00 0.232754 -0.835251 0.9998 0.0002
5 -1.00 -0.775397 0.213060 1.0000 0.0000
6 +1.00 -0.217359 -0.547669 0.0000 1.0000
7 -1.00 0.868005 -0.879819 0.9998 0.0002
8 -1.00 0.380212 -0.670712 0.9998 0.0002
9 +1.00 -0.032863 -0.799490 0.0000 1.0000
```

Where p0 and p1 are the model's output. p0 is the probability that the
target is -1 and p1 is the probability that the target is +1.


---



## Running the MLP and RNN Holdout Experiments

Be sure to download the datasets via the setup scripts.

```shell
$ ./scripts/setup.py
```

There are two experiement types that use the holdout training
method. One is a Multilayer Perceptron Model (MLP) and the other is a
Recurrent Neural Network Model (RNN).

To train the MLP model.

```shell
$ cd exprmnts/holdout-exprmnts-1/
$ train_ney.py --config=mlp-tanh.conf
$ classify_data.py --config=mlp-tanh.conf --test_datafile=test-1yr.dat --output=mlp-output.dat
```

To train the RNN model. The --time_field parameter tells classify_data.py
to organize the summary statistics by date

```shell
$ train_ney.py --config=rnn-gru-small.conf
$ classify_data.py --config=rnn-gru-small.conf --test_datafile=all-1yr.dat --time_field=date
```

