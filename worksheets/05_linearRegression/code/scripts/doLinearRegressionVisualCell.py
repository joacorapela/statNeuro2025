
import sys
import os
import argparse
import numpy as np
import pandas as pd
import scipy.stats
import skpp
import sklearn.model_selection
import plotly.subplots
import plotly.graph_objects as go

import utils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, help="results label",
                        default="simCC")
    parser.add_argument("--order", type=int, help="model order", default=2)
    parser.add_argument("--regCoef", type=float,
                        help="L2 regularization coefficient", default=0.0)
    parser.add_argument("--nRDs", type=int,
                        help="number of relevant dimensions", default=2)
    parser.add_argument("--test_percentage", type=float,
                        help="percentage of data used for testing", default=.2)
    parser.add_argument("--ciAlpha", type=float,
                        help=("significance level for coefficients bootstrap "
                              "confidence interval"), default=.05)
    parser.add_argument("--ciNResamples", type=float,
                        help=("number of resamples for coefficients bootstrap "
                              "confidence interval"), default=1000)
    parser.add_argument("--images_filename", type=str,
                        help="images filename",
                        default="http://www.gatsby.ucl.ac.uk/~rapela/neuroinformatics/2024/worksheets/linearRegression/data/equalpower_C2_25hzPP.dat")
    parser.add_argument("--responses_filename", type=str,
                        help="responses filename pattern",
                        default="http://www.gatsby.ucl.ac.uk/~rapela/neuroinformatics/2024/worksheets/linearRegression/data/ySimCC.dat")
    parser.add_argument("--figures_filename_pattern", type=str,
                        help="figures filename pattern",
                        default="../../figures/{:s}_{:s}.{:s}")
    args = parser.parse_args()

    # get command line parameters
    label = args.label
    order = args.order
    reg_coef = args.regCoef
    n_RDs = args.nRDs
    test_percentage = args.test_percentage
    ci_alpha = args.ciAlpha
    ci_nresamples = args.ciNResamples
    images_filename = args.images_filename
    responses_filename = args.responses_filename
    figures_filename_pattern = args.figures_filename_pattern

    prefix = f"{label}_order{order}_nRDs{n_RDs}_regCoef{reg_coef}"

    # create figures directory, if necessary
    fig_filename = figures_filename_pattern.format(prefix, "RDs", "png")
    dirname = os.path.dirname(fig_filename)
    if not os.path.exists(dirname):
        print(f"Creating figures directory {dirname}")
        os.mkdir(dirname)

    # get data
    images = pd.read_csv(images_filename, sep="\s+", header=None).to_numpy()
    Y = pd.read_csv(responses_filename, sep=" ", header=None).to_numpy().\
        squeeze()
    image_width = image_height = int(np.sqrt(images.shape[1]))

    # compute basis functions and projections of images onto them
    ppr = skpp.ProjectionPursuitRegressor(r=n_RDs)
    ppr_res = ppr.fit(X=images, Y=Y)
    alphas = ppr_res._alpha
    rds = np.linalg.svd(alphas)[0]
    px = np.matmul(images, rds)  # projecting images onto RDs

    # calculate regression coefficients with train data
    X = utils.buildDataMatrix(px=px, order=order, nRDs=n_RDs)
    X_train, X_test, Y_train, Y_test = \
        sklearn.model_selection.train_test_split(X, Y,
                                                 test_size=test_percentage)
    I = np.eye(X.shape[1])
    coefs = ...

    # compute coefficients bootstrap confidence intervals
    # b_coefs[i,j] is the jth coefficient obtained from the jth bootstrap
    # resample
    b_coefs = utils.bootstrapRegressionCoefs(X=X_train, Y=Y_train,
                                             reg_coef=reg_coef,
                                             n_resamples=ci_nresamples)
    b_coefs_cis = np.empty(shape=(2, len(coefs)), dtype=np.double)
    for i in range(len(coefs)):
        # b_coefs_cis[0,j] is the 95% lower coefficient for the jth coefficient
        # b_coefs_cis[1,j] is the 95% upper coefficient for the jth coefficient
        b_coefs_cis[:, i] = utils.computeBootstrapCIs(b_coefs[:, i],
                                                      alpha=ci_alpha)

    # calculate residuals
    fitted_train = ...
    residuals_train = Y_train - fitted_train

    # compute correlation coefficient on test data
    fitted_test = ...
    rho_test = np.corrcoef(Y_test, fitted_test)[0, 1]

    # Plots

    # RDs
    fig = plotly.subplots.make_subplots(rows=1, cols=n_RDs)
    for i in range(n_RDs):
        alpha = np.reshape(rds[:, i], newshape=(image_width, image_height))
        trace = go.Heatmap(z=alpha)
        fig.add_trace(trace, row=1, col=i+1)
    fig.write_image(figures_filename_pattern.format(prefix, "RDs", "png"))
    fig.write_html(figures_filename_pattern.format(prefix, "RDs", "html"))
    fig.show()

    # coefs
    fig = go.Figure()
    trace = go.Scatter(x=np.arange(len(coefs)), y=coefs,
                       error_y=dict(type="data", symmetric=False,
                                    array=b_coefs_cis[0, :]-coefs,
                                    arrayminus=coefs-b_coefs_cis[1, :]))
    fig.add_trace(trace)
    fig.add_hline(y=0)
    fig.update_layout(xaxis=dict(title="Coefficient Index"),
                      yaxis=dict(title="Coefficient Value"))
    fig.write_image(figures_filename_pattern.format(prefix, "coefs", "png"))
    fig.write_html(figures_filename_pattern.format(prefix, "coefs", "html"))
    fig.show()

    # train residuals
    x_residuals_dense = np.linspace(min(residuals_train), max(residuals_train),
                                    1000)
    mean_residuals_train = np.mean(residuals_train)
    std_residuals_train = np.std(residuals_train)
    pdf_residuals_train = scipy.stats.norm.pdf(x_residuals_dense,
                                               loc=mean_residuals_train,
                                               scale=std_residuals_train)
    fig = go.Figure()
    trace = go.Histogram(x=residuals_train, histnorm='probability density',
                         showlegend=False)
    fig.add_trace(trace)
    trace = go.Scatter(x=x_residuals_dense, y=pdf_residuals_train,
                       mode="lines", showlegend=False)
    fig.add_trace(trace)
    fig.update_layout(xaxis=dict(title="Train Residuals"),
                      yaxis=dict(title="Count"))
    fig.write_image(figures_filename_pattern.format(prefix, "residuals",
                                                    "png"))
    fig.write_html(figures_filename_pattern.format(prefix, "residuals",
                                                   "html"))
    fig.show()

    # predicted test responses
    fig = go.Figure()
    trace = go.Scatter(x=Y_test, y=fitted_test, mode="markers",
                       showlegend=False)
    fig.add_trace(trace)
    trace = go.Scatter(x=Y_test, y=Y_test, mode="lines",
                       line=dict(color="gray"), showlegend=False)
    fig.add_trace(trace)
    fig.update_layout(xaxis=dict(title="Test Responses"),
                      yaxis=dict(title="Predicted Test Responses"),
                      title=f"r={rho_test:.05f}")
    fig.write_image(figures_filename_pattern.format(prefix, "predictions",
                                                    "png"))
    fig.write_html(figures_filename_pattern.format(prefix, "predictions",
                                                   "html"))
    fig.show()

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
