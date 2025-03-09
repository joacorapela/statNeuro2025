import sys
import os
import argparse
import pickle
import numpy as np
import torch
import utils
import myNets

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_hidden", help="number of units in the hidden layer",
                        type=int, default=200)
    parser.add_argument("--prop_train", help="data proportion for training",
                        type=float, default=0.6)
    parser.add_argument("--optim_type", help="optimizer type (SGD, LBFGS)",
                        type=str, default="SGD")
    parser.add_argument("--n_epochs", help="number of epochs", type=int,
                        default=2000)
    parser.add_argument("--learning_rate", help="learning rate", type=float,
                        default=1e-5)
    parser.add_argument("--train_loss_fn_type",
                        help="type of train loss function (MSE or Circular)", type=str,
                        default="Circular")
    parser.add_argument("--test_loss_fn_type",
                        help="type of test loss function (MSE or Circular)", type=str,
                        default="Circular")
    parser.add_argument("--random_seed", help="random seed", type=int, default=4)
    parser.add_argument("--results_filename_pattern", type=str,
                        help="results filename pattern",
                        default="results/nn_nHidden{:d}_propTrain{:.2f}_nEpochs{:d}_optimType{:s}_learningRate{:f}_trainLossFn_{:s}_testLossFn{:s}_randomSeed{:d}.{:s}")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="figures/trainTestLosses_nHidden{:d}_propTrain{:.2f}_nEpochs{:d}_optimType{:s}_learningRate{:f}_trainLossFn_{:s}_testLosFn{:s}_randomSeed{:d}.{:s}")
    args = parser.parse_args()

    n_hidden = args.n_hidden
    prop_train = args.prop_train
    optim_type = args.optim_type
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    train_loss_fn_type = args.train_loss_fn_type
    test_loss_fn_type = args.test_loss_fn_type
    random_seed = args.random_seed
    results_filename_pattern = args.results_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern

    results_filename = results_filename_pattern.format(n_hidden, prop_train,
                                                       n_epochs, optim_type,
                                                       learning_rate,
                                                       train_loss_fn_type,
                                                       test_loss_fn_type,
                                                       random_seed, "pickle")
    png_fig_filename = fig_filename_pattern.format(n_hidden, prop_train,
                                                   n_epochs, optim_type,
                                                   learning_rate,
                                                   train_loss_fn_type,
                                                   test_loss_fn_type,
                                                   random_seed, "png")
    html_fig_filename = fig_filename_pattern.format(n_hidden, prop_train,
                                                    n_epochs, optim_type,
                                                    learning_rate,
                                                    train_loss_fn_type,
                                                    test_loss_fn_type,
                                                    random_seed, "html")
    dirname = os.path.dirname(png_fig_filename)
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    with open(results_filename, "rb") as f:
        load_res = pickle.load(f)

    train_loss = load_res["train_loss"]
    test_loss = load_res["test_loss"]

    fig = utils.get_fig_train_test_loss(train_loss=train_loss, test_loss=test_loss)
    fig.write_image(png_fig_filename)
    fig.write_html(html_fig_filename)

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
