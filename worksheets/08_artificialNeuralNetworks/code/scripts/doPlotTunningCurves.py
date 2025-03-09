import sys
import argparse
import pickle
import numpy as np
import plotly.graph_objects as go
import utils

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank_1st_neuron",
                        help=("rank of 1st neuron to plot "
                              "(neurons sorted in decreasing order by |w|)"),
                        type=int, default=0)
    parser.add_argument("--n_neurons_to_plot", help="number of neurons to plot",
                        type=int, default=50)
    parser.add_argument("--n_hidden", help="number of units in the hidden layer",
                        type=int, default=200)
    parser.add_argument("--prop_train", help="data proportion for training",
                        type=float, default=0.6)
    parser.add_argument("--optim_type", help="optimizer type (SGD, LBFGS)",
                        type=str, default="SGD")
    parser.add_argument("--n_epochs", help="number of epochs", type=int,
                        default=20000)
    parser.add_argument("--learning_rate", help="learning rate", type=float,
                        default=1e-2)
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
                        default="figures/tunningCurves_rank1stNeuron{:d}_nNeurons{:d}_nHidden{:d}_propTrain{:.2f}_nEpochs{:d}_optimType{:s}_learningRate{:f}_trainLossFn_{:s}_testLossFn{:s}_randomSeed{:d}.{:s}")
    parser.add_argument("--data_filename", help="data filename", type=str,
                        default="data/W3D4_stringer_oribinned1.npz")
    parser.add_argument("--url", help="data url", type=str,
                        default="https://osf.io/683xc/download")
    parser.add_argument("--expected_md5", help="expected md5", type=str,
                        default="436599dfd8ebe6019f066c38aed20580")
    args = parser.parse_args()

    rank_1st_neuron = args.rank_1st_neuron
    n_neurons_to_plot = args.n_neurons_to_plot
    n_hidden = args.n_hidden
    prop_train = args.prop_train
    optim_type = args.optim_type
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    train_loss_fn_type = args.train_loss_fn_type
    test_loss_fn_type = args.test_loss_fn_type
    random_seed = args.random_seed
    results_filename_pattern = args.results_filename_pattern
    data_filename = args.data_filename
    url = args.url
    expected_md5 = args.expected_md5
    fig_filename_pattern = args.fig_filename_pattern

    resp_all, stimuli_all = utils.load_data(data_filename=data_filename,
                                            url=url, expected_md5=expected_md5)
    _, n_neurons = resp_all.shape

    results_filename = results_filename_pattern.format(n_hidden, prop_train,
                                                       n_epochs, optim_type,
                                                       learning_rate,
                                                       train_loss_fn_type,
                                                       test_loss_fn_type,
                                                       random_seed, "pickle")

    with open(results_filename, "rb") as f:
        load_res = pickle.load(f)

    net = load_res["net"]
    weights = net.in_layer.weight.detach().numpy()
    l2_norms = np.linalg.norm(weights, ord=2, axis=0)
    sort_indices = np.argsort(-l2_norms)
    neuron_ids = np.arange(n_neurons)

    neuron_ids_to_plot = neuron_ids[sort_indices][rank_1st_neuron:(rank_1st_neuron+n_neurons_to_plot)]

    fig = go.Figure()
    for neuron_id in neuron_ids_to_plot:
        trace = go.Scatter(y=resp_all[:, neuron_id], mode="lines+markers",
                           name=f"neuron {neuron_id}")
        fig.add_trace(trace)
    fig.update_xaxes(title="Orientation (degrees)")
    fig.update_yaxes(title="Fluorescence")

    png_fig_filename = fig_filename_pattern.format(rank_1st_neuron,
                                                   n_neurons_to_plot,
                                                   n_hidden, prop_train,
                                                   n_epochs, optim_type,
                                                   learning_rate,
                                                   train_loss_fn_type,
                                                   test_loss_fn_type,
                                                   random_seed,
                                                   "png")
    html_fig_filename = fig_filename_pattern.format(rank_1st_neuron,
                                                    n_neurons_to_plot,
                                                    n_hidden, prop_train,
                                                    n_epochs, optim_type,
                                                    learning_rate,
                                                    train_loss_fn_type,
                                                    test_loss_fn_type,
                                                    random_seed,
                                                    "html")
    fig.write_image(png_fig_filename)
    fig.write_html(html_fig_filename)
    fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
