import sys
import argparse
import os
import numpy as np
import scipy.stats
import plotly.graph_objs as go

import one.api
import brainbox.io.one

import utils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_size", type=float, help="bin_size", default=1.0)
    parser.add_argument("--experiment_id", type=str,
                        help="experiment to analyze",
                        default="ebe2efe3-e8a1-451a-8947-76ef42427cc9")
    parser.add_argument("--probe_id", type=str,
                        help="id of the probe to analyze",
                        default="probe00")
    parser.add_argument("--x_label", type=str, help="x_label",
                        default="Time (sec)")
    parser.add_argument("--y_label", type=str, help="x_label",
                        default="Neuron ID")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default=("figures/binned_spikes_svd_"
                                 "binSize_{:.02f}_{:s}.{:s}"))
    args = parser.parse_args()

    bin_size = args.bin_size
    eID = args.experiment_id
    probe_id = args.probe_id
    x_label = args.x_label
    y_label = args.y_label
    x_label = args.x_label
    fig_filename_pattern = args.fig_filename_pattern

    aOne = one.api.ONE(base_url='https://openalyx.internationalbrainlab.org',
                       password='international', silent=True)
    spikes = aOne.load_object(eID, 'spikes', f'alf/{probe_id}/pykilosort')
    clusters = aOne.load_object(eID, "clusters", f"alf/{probe_id}/pykilosort")
    trials = aOne.load_object(eID, 'trials')
    els = brainbox.io.one.load_channel_locations(eID, one=aOne)

    # bin and z-score
    spike_time_binary = np.floor(spikes.times/bin_size).astype(int)
    activity_array, edges = np.histogramdd(
        (spike_time_binary, spikes.clusters),
        bins=(spike_time_binary.max(), spikes.clusters.max()))
    # after transposing the shape of activity_array is n_neurons x n_times
    activity_arrayTZ = scipy.stats.zscore(activity_array.T, axis=1)

    u, s, vh = np.linalg.svd(a=activity_arrayTZ, full_matrices=False)

    times = (edges[0][1:] + edges[0][:-1])/2.0*bin_size
    neurons_ids = edges[1][:-1]

    zmin, zmax = np.percentile(activity_arrayTZ, q=(1.0, 99.0))

    # sorted z-scores
    sorted_neurons_indices = np.argsort(?[:, 0])
    activity_arrayZ_sorted = ???
    hovertext = utils.getHovertext(
        times=times, neurons_ids=neurons_ids, z=activity_arrayZ_sorted,
        channels_for_neurons=clusters.channels,
        regions_for_channels=els[probe_id]["acronym"])
    fig = utils.getHeatmap(xs=times, ys=neurons_ids,
                           zs=activity_arrayZ_sorted,
                           hovertext=hovertext, zmin=zmin, zmax=zmax,
                           x_label=x_label, y_label=y_label)

    # create figure directory if it does not exist
    fig_filename = fig_filename_pattern.format(bin_size, "u0Sorted",  "png")
    dirname = os.path.dirname(fig_filename)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    #

    fig.add_vline(x=max(trials.response_times))
    fig.write_image(fig_filename_pattern.format(bin_size, "u0Sorted",  "png"))
    fig.write_html(fig_filename_pattern.format(bin_size, "u0Sorted",  "html"))

    fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
