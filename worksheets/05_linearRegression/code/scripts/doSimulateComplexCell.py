
import sys
import os
import argparse
import numpy as np
import pandas as pd
import plotly.subplots
import plotly.graph_objects as go

import utils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr", type=float,
                        help="responses snr",
                        default=1.0)
    parser.add_argument("--images_filename", type=str,
                        help="images filename",
                        default="http://www.gatsby.ucl.ac.uk/~rapela/neuroinformatics/2024/worksheets/linearRegression/data/equalpower_C2_25hzPP.dat")
    parser.add_argument("--responses_filename", type=str,
                        help="simulated responses filename",
                        default="../../results/ySimCC.dat")
    parser.add_argument("--figures_filename_pattern", type=str,
                        help="figures filename pattern",
                        default="../../figures/simCC_{:s}.{:s}")
    args = parser.parse_args()

    # get command line parameters
    snr = args.snr
    images_filename = args.images_filename
    responses_filename = args.responses_filename
    figures_filename_pattern = args.figures_filename_pattern

    # create figures and results directory, if necessary
    dirname = os.path.dirname(responses_filename)
    if not os.path.exists(dirname):
        print(f"Creating figures directory {dirname}")
        os.mkdir(dirname)
    fig_filename = figures_filename_pattern.format("RDs", "png")
    dirname = os.path.dirname(fig_filename)
    if not os.path.exists(dirname):
        print(f"Creating figures directory {dirname}")
        os.mkdir(dirname)

    # get data
    images = pd.read_csv(images_filename, sep="\s+", header=None).to_numpy()
    image_width = image_height = int(np.sqrt(images.shape[1]))

    # compute basis functions and projections of images onto them
    gabor0 = utils.gabor(sigma=2.5, theta=-np.pi/4, Lambda=1, psi=0,
                         gamma=1)
    gabor0 = gabor0[:12, :12].flatten()
    gabor1 = utils.gabor(sigma=2.5, theta=-np.pi/4, Lambda=1, psi=np.pi/2,
                         gamma=1)
    gabor1 = gabor1[:12, :12].flatten()

    rds = np.zeros(shape=(images.shape[1], 2), dtype=np.double)
    rds[:len(gabor0), 0] = gabor0.flatten()
    rds[:len(gabor1), 1] = gabor1.flatten()

    # simulate responses
    noiseless_responses = (images @ rds[:, 0])**2 + (images @ rds[:, 1])**2

    max_firing_rate = 100
    noiseless_responses *= max_firing_rate / np.max(noiseless_responses)

    # get noise from snr
    std_noise = np.mean(noiseless_responses) / snr
    noise = np.random.normal(loc=0, scale=std_noise, size=images.shape[0])

    responses = noiseless_responses + noise
    # responses[responses < 0] = 0.0

    # save responses
    df = pd.DataFrame(responses)
    df.to_csv(responses_filename, header=None, index=False)
    print(f"Simulation written to {responses_filename}")

    # plot
    # RDs
    fig = plotly.subplots.make_subplots(rows=1, cols=2)
    for i in range(2):
        alpha = np.reshape(rds[:, i], newshape=(image_width, image_height))
        trace = go.Heatmap(z=alpha)
        fig.add_trace(trace, row=1, col=i+1)
    fig.write_image(figures_filename_pattern.format("rds", "png"))
    fig.write_html(figures_filename_pattern.format("rds", "html"))
    fig.show()

    # responses
    fig = go.Figure()
    trace = go.Scatter(x=np.arange(len(responses)), y=responses,
                       mode="markers", name="responses")
    fig.add_trace(trace)
    trace = go.Scatter(x=np.arange(len(noiseless_responses)),
                       y=noiseless_responses,
                       mode="markers", name="noiseless_responses")
    fig.add_trace(trace)
    fig.update_layout(xaxis=dict(title="Index"), yaxis=dict(title="Response"))
    fig.write_image(figures_filename_pattern.format("responses", "png"))
    fig.write_html(figures_filename_pattern.format("responses", "html"))
    fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
