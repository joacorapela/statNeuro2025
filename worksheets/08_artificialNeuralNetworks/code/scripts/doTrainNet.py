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
    parser.add_argument("--data_filename", help="data filename", type=str,
                        default="data/W3D4_stringer_oribinned1.npz")
    parser.add_argument("--url", help="data url", type=str,
                        default="https://osf.io/683xc/download")
    parser.add_argument("--expected_md5", help="expected md5", type=str,
                        default="436599dfd8ebe6019f066c38aed20580")
    parser.add_argument("--n_hidden", help="number of units in the hidden layer",
                        type=int, default=10)
    parser.add_argument("--prop_train", help="data proportion for training",
                        type=float, default=0.6)
    parser.add_argument("--optim_type", help="optimizer type (SGD, LBFGS)",
                        type=str, default="SGD")
    parser.add_argument("--n_epochs", help="number of epochs", type=int, default=50)
    parser.add_argument("--learning_rate", help="learning rate", type=float,
                        default=1e-4)
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
    args = parser.parse_args()

    data_filename = args.data_filename
    url = args.url
    expected_md5 = args.expected_md5
    n_hidden = args.n_hidden
    prop_train = args.prop_train
    optim_type = args.optim_type
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    train_loss_fn_type = args.train_loss_fn_type
    test_loss_fn_type = args.test_loss_fn_type
    random_seed = args.random_seed
    results_filename_pattern = args.results_filename_pattern

    resp_all, stimuli_all = utils.load_data(data_filename=data_filename,
                                            url=url, expected_md5=expected_md5)
    n_stimuli, n_neurons = resp_all.shape

    if random_seed > 0:
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    # Split data into training set and testing set
    n_train = int(prop_train * n_stimuli)
    ishuffle = torch.randperm(n_stimuli)
    itrain = ishuffle[:n_train]  # indices of data samples to include in training set
    itest = ishuffle[n_train:]  # indices of data samples to include in testing set
    stimuli_test = stimuli_all[itest]
    resp_test = resp_all[itest]
    stimuli_train = stimuli_all[itrain]
    resp_train = resp_all[itrain]

    # Initialize network with 10 hidden units
    net = myNets.DeepNetReLU(n_neurons, n_hidden)

    if train_loss_fn_type == "MSE":
        train_loss_fn = torch.nn.MSELoss()
    elif train_loss_fn_type == "Circular":
        train_loss_fn = myNets.CircularLoss()
    else:
        raise InvalidArgumentException(
            f"Invalid train_loss_fn_type={train_loss_fn_type}"
        )

    if test_loss_fn_type == "MSE":
        test_loss_fn = torch.nn.MSELoss()
    elif test_loss_fn_type == "Circular":
        test_loss_fn = myNets.CircularLoss()
    else:
        raise InvalidArgumentException(
            f"Invalid test_loss_fn_type={test_loss_fn_type}"
        )

    if optim_type == "SGD":
        # Initialize PyTorch SGD optimizer
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    elif optim_type == "Adam":
        # Initialize PyTorch Adam optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    else:
        raise InvalidArgumentException(
            f"Invalid optim_type={optim_type}"
        )

    # Move network and data to GPU, if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running network on {device}")
    net.to(device)
    stimuli_train = stimuli_train.to(device)
    resp_train = resp_train.to(device)
    stimuli_test = stimuli_test.to(device)
    resp_test = resp_test.to(device)

    # Run gradient descent on data
    train_loss, test_loss = myNets.train(net=net, optimizer=optimizer,
                                         train_loss_fn=train_loss_fn,
                                         test_loss_fn=test_loss_fn,
                                         train_data=resp_train,
                                         train_labels=stimuli_train,
                                         test_data=resp_test,
                                         test_labels=stimuli_test,
                                         n_epochs=n_epochs)

    # save results
    results_filename = results_filename_pattern.format(n_hidden, prop_train,
                                                       n_epochs, optim_type,
                                                       learning_rate,
                                                       train_loss_fn_type,
                                                       test_loss_fn_type,
                                                       random_seed, "pickle")
    dirname = os.path.dirname(results_filename)
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    results = dict(net=net, train_loss=train_loss, test_loss=test_loss,
                   stimuli_train=stimuli_train, stimuli_test=stimuli_test)
    with open(results_filename, "wb") as f:
        pickle.dump(results, f)

    # Plot the training loss over iterations of GD
    # fig = utils.get_fig_train_test_loss(train_loss=train_loss, test_loss=test_loss)
    # fig.show()

    breakpoint()

if __name__ == "__main__":
    main(sys.argv)
