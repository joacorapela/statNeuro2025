#!/bin/csh

ipython --pdb doPlotTrainTestLosses.py -- --n_hidden=200 --n_epochs=20000 --learning_rate=1e-5 --train_loss_fn_type=MSE --test_loss_fn_type=MSE --optim_type=SGD
