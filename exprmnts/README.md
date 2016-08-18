# Experiments for DNN Quant

All experiments for the dnn-quant project are located in sub directories of this directory

```shell
$ cd system-tests
$ train_net.py --config=system-test.conf
$ classify_data.py --config=system-test.conf > preds.dat
$ paste -d ' ' $DNN_QUANT_ROOT/datasets/mlp-xor-test.dat preds.dat  > results.dat
$ head results.dat
```
