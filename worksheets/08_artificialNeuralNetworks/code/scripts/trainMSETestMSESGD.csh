#!/bin/csh

ipython --pdb doTrainNet.py -- --n_hidden=200 --n_epochs=20000 --learning_rate=1e-5 --train_loss_fn_type=MSE --test_loss_fn_type=MSE --optimizer_type=SGD
