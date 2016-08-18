# Experiments for DNN Quant

All experiments for the dnn-quant project are located in sub
directories of this directory.

```shell
$ cd system-tests
$ train_net.py --config=system-test.conf
$ classify_data.py --config=system-test.conf > preds.dat
$ paste -d ' ' $DNN_QUANT_ROOT/datasets/mlp-xor-test.dat preds.dat  > results.dat
$ head results.dat
```

Which should output a space seperated file that looks like:

```shell
id target x1 x2 p0 p1
0 +1.00 -0.446373 -0.715276 0.0000 1.0000
0 +1.00 0.149692 0.896433 0.0000 1.0000
0 +1.00 -0.803404 -0.377976 0.0000 1.0000
0 -1.00 0.232754 -0.835251 0.9998 0.0002
0 -1.00 -0.775397 0.213060 1.0000 0.0000
0 +1.00 -0.217359 -0.547669 0.0000 1.0000
0 -1.00 0.868005 -0.879819 0.9998 0.0002
0 -1.00 0.380212 -0.670712 0.9998 0.0002
0 +1.00 -0.032863 -0.799490 0.0000 1.0000
```

Where p0 and p1 are the model's output. p0 is the probability that the
target is -1 and p1 is the probability that the target is +1.
