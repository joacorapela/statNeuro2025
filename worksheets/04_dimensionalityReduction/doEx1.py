import sys
import argparse
import os
import numpy as np
import scipy.stats

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
    parser.add_argument("--colorbar_title", type=str, help="colorbar_title",
                        default="Z-scored<br>spike rate")
    parser.add_argument("--fig_filename_pattern", type=str,
                        help="figure filename pattern",
                        default=("figures/binned_spikes_binSize_{:.02f}_"
                                 "{:s}.{:s}"))
    args = parser.parse_args()

    bin_size = args.bin_size
    eID = args.experiment_id
    probe_id = args.probe_id
    x_label = args.x_label
    y_label = args.y_label
    colorbar_title = args.colorbar_title
    fig_filename_pattern = args.fig_filename_pattern

    aOne = one.api.ONE(base_url='https://openalyx.internationalbrainlab.org',
                       password='international', silent=True)
    spikes = aOne.load_object(eID, 'spikes', f'alf/{probe_id}/pykilosort')
    clusters = aOne.load_object(eID, "clusters", f"alf/{probe_id}/pykilosort")
    els = brainbox.io.one.load_channel_locations(eID, one=aOne)

    spike_time_binary = np.floor(spikes.times/bin_size).astype(int)
    activity_array, edges = np.histogramdd(
        (spike_time_binary, spikes.clusters),
        bins=(spike_time_binary.max(), spikes.clusters.max()))
    # after transposing the shape of activity_array is n_neurons x n_times
    activity_arrayTZ = scipy.stats.zscore(activity_array.T, axis=1)

    times = (edges[0][1:] + edges[0][:-1])/2.0
    neurons_ids = edges[1][:-1]

    zmin, zmax = np.percentile(activity_arrayTZ, q=(1.0, 99.0))

    hovertext = utils.getHovertext(
        times=times, neurons_ids=neurons_ids, z=activity_arrayTZ,
        channels_for_neurons=clusters.channels,
        regions_for_channels=els[probe_id]["acronym"])

    fig = utils.getHeatmap(xs=times, ys=neurons_ids, zs=activity_arrayTZ,
                           hovertext=hovertext, zmin=zmin, zmax=zmax,
                           x_label=x_label, y_label=y_label,
                           colorbar_title=colorbar_title)

    # create figure directory if it does not exist
    fig_filename = fig_filename_pattern.format(bin_size, "original",  "png")
    dirname = os.path.dirname(fig_filename)
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    fig.write_image(fig_filename_pattern.format(bin_size, "original",  "png"))
    fig.write_html(fig_filename_pattern.format(bin_size, "original",  "html"))
    fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
