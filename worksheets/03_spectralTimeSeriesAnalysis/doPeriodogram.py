import sys
import argparse
import numpy as np
import scipy.io
import scipy.fft

import plotly.graph_objects as go


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_factor", type=int,
                        help="downsample factor", default=10)
    parser.add_argument("--channel_label", type=str, help="channels label",
                        default="Amygdala1")
    parser.add_argument("--data_filename_pattern", type=str,
                        help="data filename pattern",
                        default="../../data/time_data_pre_45sec_ds{:d}_v6.mat")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="../../figures/{:s}_time_data_pre_45sec_ds{:d}_channel{:s}_v6.{:s}")
    args = parser.parse_args()

    ds_factor = args.ds_factor
    channel_label = args.channel_label
    data_filename_pattern = args.data_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern

    data_filename = data_filename_pattern.format(ds_factor)
    mat = scipy.io.loadmat(data_filename)
    data = mat["time_data"]["trial"][0,0][0,0]
    times = mat["time_data"]["time"][0,0][0,0][0,:]
    fsample = mat["time_data"]["fsample"][0,0][0,0]

    n_channels = 8

    # get channel labels
    chanlabels = [None] * n_channels
    for i in range(n_channels):
        chanlabels[i] = mat["time_data"]["label"][0,0][i,0][0]

    channel_index = chanlabels.index(channel_label)
    channel_data = data[channel_index, :]

    df = ...
    fNQ = ...

    # estimate spectral density
    N = data.shape[1]
    if N % 2 == 0:
        pos_indices = slice(0, N//2-1, 1)
        freqs = np.arange(N//2-1) * df
    else:
        pos_indices = slice(0, (N-1)//2, 1)
        freqs = np.arange((N-1)/2) * df
    channel_data_fft = scipy.fft.fft(channel_data)[pos_indices]
    channel_psd = ...

    # plot
    fig = go.Figure()
    trace = go.Scatter(x=freqs, y=channel_psd)
    fig.add_trace(trace)
    fig.update_layout(xaxis_title="Frequency (Hz)",
                      yaxis_title="Power Spectrum")
    fig.write_image(fig_filename_pattern.format("periodogram", ds_factor, channel_label, "png"))
    fig.write_html(fig_filename_pattern.format("periodogram", ds_factor, channel_label, "html"))
    fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
