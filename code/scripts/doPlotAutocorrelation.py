
import sys
import argparse
import numpy as np
import scipy.io
import plotly.graph_objects as go


def main(argv):
    parser = argparse.ArgumentParser()
    # parser.add_argument("--trial_index", type=int, help="trial index",
    #                     default=0)
    parser.add_argument("--channel_label", type=str, help="channels label",
                        default="Cz")
    parser.add_argument("--max_lag", type=int, help="maximum lag", default=200)
    parser.add_argument("--diff_n", type=int,
                        help="number of times n fore differentation",
                        default=0)
    parser.add_argument("--data_field", type=str,
                        help="data field inside the mat file", default="data")
    parser.add_argument("--data_filename", type=str, help="data filename",
                        default="../../../data/D_01_cleaned.mat")
    args = parser.parse_args()

    # trial_index = args.trial_index
    channel_label = args.channel_label
    max_lag = args.max_lag
    diff_n = args.diff_n
    data_field = args.data_field
    data_filename = args.data_filename

    mat = scipy.io.loadmat(data_filename)
    srate = mat["srate"].squeeze()
    data = mat[data_field]
    # times = mat["times"]
    n_channels = len(mat["chanlabels"][0])

    N = data.shape[1]
    lags = np.arange(max_lag)
    lags_sec = lags.astype(float)/srate
    # get channel labels
    chanlabels = [None] * n_channels
    for i in range(n_channels):
        chanlabels[i] = mat["chanlabels"][0][i][0]

    channel_index = chanlabels.index(channel_label)

    n_trials = data.shape[2]
    cor = np.empty((len(lags), n_trials), dtype=np.double)
    fig = go.Figure()
    for r in range(n_trials):
        channel_data = data[channel_index, :, r]
        channel_data_diff = np.diff(channel_data, n=diff_n)
        mu_hat = channel_data_diff.mean()
        var_hat = channel_data_diff.var()
        for lag in lags:
            if lag == 0:
                cor[lag, r] = 1.0
            else:
                cor[lag, r] = np.mean(
                    (channel_data_diff[lag:]-mu_hat) *
                    (channel_data_diff[:-lag]-mu_hat))/var_hat

        trace = go.Scatter(x=lags_sec, y=cor[:, r], mode="lines+markers",
                           name=f"trial {r}")
        fig.add_trace(trace)
    fig.add_hline(y=1.0/np.sqrt(N), line_dash="dash", line_color="gray")
    fig.add_hline(y=-1.0/np.sqrt(N), line_dash="dash", line_color="gray")
    fig.update_layout(xaxis_title="Lag (sec)", yaxis_title="Autocorrelation")
    fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
