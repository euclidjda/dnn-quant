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
$ cd exprmnts/system-test
$ train_net.py --config=system-test.conf
$ classify_data.py --config=system-test.conf > test.out 
```

### Tested platforms include:
- python 3.4, Ubuntu 14.04.5 LTS, with GPU, tensorflow r0.10 & tensorflow r0.08
- python 3.5, Mac OSX 10.11.5, no GPU, tensorflow r0.10 & tensorflow r0.08

