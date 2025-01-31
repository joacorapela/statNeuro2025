import sys
import argparse
import numpy as np
import scipy.io
import plotly.graph_objects as go


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_factor", type=int,
                        help="downsample factor", default=10)
    parser.add_argument("--channel_label", type=str, help="channel label",
                        default="Amygdala1")
    parser.add_argument("--plot_start_time", type=float,
                        help="plot start time (sec)", default=0.0)
    parser.add_argument("--plot_end_time", type=float,
                        help="plot end time (sec)", default=20.0)
    parser.add_argument("--data_filename_pattern", type=str,
                        help="data filename pattern",
                        default="../../data/time_data_pre_45sec_ds{:d}_v6.mat")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default="../../figures/time_data_pre_45sec_ds{:d}_channel{:s}_startTime{:.02f}_end_time{:.02f}.{:s}")
    args = parser.parse_args()

    ds_factor = args.ds_factor
    plot_start_time = args.plot_start_time
    plot_end_time = args.plot_end_time
    channel_label = args. channel_label
    data_filename_pattern = args.data_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern

    data_filename = data_filename_pattern.format(ds_factor)
    mat = scipy.io.loadmat(data_filename)
    data = mat["time_data"]["trial"][0,0][0,0]
    times = mat["time_data"]["time"][0,0][0,0][0,:]
    n_channels = 8

    plot_start_index = np.where(times>=plot_start_time)[0][0]
    plot_end_index = np.where(times>=plot_end_time)[0][0]
    # get channel labels
    chanlabels = [None] * n_channels
    for i in range(n_channels):
        chanlabels[i] = mat["time_data"]["label"][0,0][i,0][0]

    channel_index = chanlabels.index(channel_label)

    fig = go.Figure()
    channel_data = data[channel_index, :]
    trace = go.Scatter(x=times[plot_start_index:plot_end_index],
                               y=channel_data[plot_start_index:plot_end_index],
                       mode="lines+markers")
    fig.add_trace(trace)
    fig.update_layout(xaxis_title="Time (msec)", yaxis_title="Voltage")
    fig.show()

    fig.write_image(fig_filename_pattern.format(ds_factor, channel_label,
                                                plot_start_time, plot_end_time,
                                                "png"))
    fig.write_html(fig_filename_pattern.format(ds_factor, channel_label,
                                               plot_start_time, plot_end_time,
                                               "html"))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
