import sys
import argparse
import os
import numpy as np
import scipy.stats

import one.api
import brainbox.io.one

import utils


def truncate_svd(u, s, vh, n_components):
    truncated_svd = np.zeros((u.shape[0], vh.shape[1]), dtype=u.dtype)
    for i in range(n_components):
        truncated_svd += ???
    return truncated_svd


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_components", type=int,
                        help="number of SVD components to keep", default=1)
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
                        default=("figures/binned_spikes_truncatedSVD_"
                                 "binSize_{:.02f}_nComponetns_{:d}.{:s}"))
    args = parser.parse_args()

    n_components = args.n_components
    bin_size = args.bin_size
    eID = args.experiment_id
    probe_id = args.probe_id
    x_label = args.x_label
    y_label = args.y_label
    x_label = args.x_label
    y_label = args.y_label
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

    times = (edges[0][1:] + edges[0][:-1])/2.0*bin_size
    neurons_ids = edges[1][:-1]

    u, s, vh = np.linalg.svd(a=activity_arrayTZ, full_matrices=False)

    sorted_neurons_indices = np.argsort(u[:, 0])

    # z_truncated = truncate_svd(u=u, s=s, vh=vh[:, argsort_res],
    z_truncated = truncate_svd(u=u[sorted_neurons_indices, :], s=s, vh=vh,
                               n_components=n_components)

    zmin, zmax = np.percentile(activity_arrayTZ, q=(1.0, 99.0))

    empiricalError = np.linalg.norm(
        # z_truncated-activity_arrayTZ, ord="fro")
        z_truncated-activity_arrayTZ[sorted_neurons_indices, :], ord="fro")
    analyticalError = np.sqrt(np.power(s[n_components:], 2).sum())

    title = (f"analytical error: {analyticalError:.2f}, "
             f"empirical error: {empiricalError:.2f}")
    print(title)
    hovertext = utils.getHovertext(
        times=times, neurons_ids=neurons_ids, z=z_truncated,
        channels_for_neurons=clusters.channels,
        regions_for_channels=els[probe_id]["acronym"])

    fig = utils.getHeatmap(xs=times, ys=neurons_ids, zs=z_truncated,
                           hovertext=hovertext, zmin=zmin, zmax=zmax,
                           x_label=x_label, y_label=y_label, title=title)

    # create figure directory if it does not exist
    fig_filename = fig_filename_pattern.format(bin_size, n_components,  "png")
    dirname = os.path.dirname(fig_filename)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    #

    fig.write_image(fig_filename_pattern.format(bin_size, n_components, "png"))
    fig.write_html(fig_filename_pattern.format(bin_size, n_components, "html"))

    # fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
