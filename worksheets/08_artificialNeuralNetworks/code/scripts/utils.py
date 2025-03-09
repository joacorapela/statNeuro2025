import os
import requests
import numpy as np
import torch
import plotly.graph_objects as go


def load_data(data_filename, url, expected_md5, bin_width=1):
    """Load mouse V1 data from Stringer et al. (2019)

    Data from study reported in this preprint:
    https://www.biorxiv.org/content/10.1101/679324v2.abstract

    These data comprise time-averaged responses of ~20,000 neurons
    to ~4,000 stimulus gratings of different orientations, recorded
    through Calcium imaging. The responses have been normalized by
    spontaneous levels of activity and then z-scored over stimuli, so
    expect negative numbers. They have also been binned and averaged
    to each degree of orientation.

    This function returns the relevant data (neural responses and
    stimulus orientations) in a torch.Tensor of data type torch.float32
    in order to match the default data type for nn.Parameters in
    Google Colab.

    This function will actually average responses to stimuli with orientations
    falling within bins specified by the bin_width argument. This helps
    produce individual neural "responses" with smoother and more
    interpretable tuning curves.

    Args:
      bin_width (float): size of stimulus bins over which to average neural
        responses

    Returns:
      resp (torch.Tensor): n_stimuli x n_neurons matrix of neural responses,
          each row contains the responses of each neuron to a given stimulus.
          As mentioned above, neural "response" is actually an average over
          responses to stimuli with similar angles falling within specified bins.
      stimuli: (torch.Tensor): n_stimuli x 1 column vector with orientation
          of each stimulus, in degrees. This is actually the mean orientation
          of all stimuli in each bin.

    """
    if not os.path.isfile(data_filename):
        try:
            r = requests.get(url)
        except requests.ConnectionError:
            print("!!! Failed to download data !!!")
        else:
            if r.status_code != requests.codes.ok:
                print("!!! Failed to download data !!!")
            elif hashlib.md5(r.content).hexdigest() != expected_md5:
                print("!!! Data download appears corrupted !!!")
            else:
                with open(data_filename, "wb") as fid:
                    fid.write(r.content)

    with np.load(data_filename) as dobj:
        data = dict(**dobj)
    resp = data['resp']
    stimuli = data['stimuli']

    if bin_width > 1:
        # Bin neural responses and stimuli
        bins = np.digitize(stimuli, np.arange(0, 360 + bin_width, bin_width))
        stimuli_binned = np.array([stimuli[bins == i].mean() for i in np.unique(bins)])
        resp_binned = np.array([resp[bins == i, :].mean(0) for i in np.unique(bins)])
    else:
        resp_binned = resp
        stimuli_binned = stimuli

    # Return as torch.Tensor
    resp_tensor = torch.tensor(resp_binned, dtype=torch.float32)
    stimuli_tensor = torch.tensor(stimuli_binned, dtype=torch.float32).unsqueeze(1)  # add singleton dimension to make a column vector

    return resp_tensor, stimuli_tensor


def get_data(n_stim, train_data, train_labels):
    """ Return n_stim randomly drawn stimuli/resp pairs

    Args:
      n_stim (scalar): number of stimuli to draw
      resp (torch.Tensor):
      train_data (torch.Tensor): n_train x n_neurons tensor with neural
        responses to train on
      train_labels (torch.Tensor): n_train x 1 tensor with orientations of the
        stimuli corresponding to each row of train_data, in radians

    Returns:
      (torch.Tensor, torch.Tensor): n_stim x n_neurons tensor of neural responses and n_stim x 1 of orientations respectively
    """
    n_stimuli = train_labels.shape[0]
    istim = np.random.choice(n_stimuli, n_stim)
    r = train_data[istim]  # neural responses to this stimulus
    ori = train_labels[istim]  # true stimulus orientation

    return r, ori


# @title Plotting Functions

def plot_data_matrix(X, ax, show=False):
    """Visualize data matrix of neural responses using a heatmap

    Args:
      X (torch.Tensor or np.ndarray): matrix of neural responses to visualize
          with a heatmap
      ax (matplotlib axes): where to plot
      show (boolean): enable plt.show()

    """

    cax = ax.imshow(X, cmap=mpl.cm.pink, vmin=np.percentile(X, 1),
                    vmax=np.percentile(X, 99))
    cbar = plt.colorbar(cax, ax=ax, label='normalized neural response')

    ax.set_aspect('auto')
    ax.set_xticks([])
    ax.set_yticks([])
    if show:
      plt.show()


def plot_train_loss(train_loss):
    plt.plot(train_loss)
    plt.xlim([0, None])
    plt.ylim([0, None])
    plt.xlabel('iterations of gradient descent')
    plt.ylabel('mean squared error')
    plt.show()


def get_fig_train_test_loss(train_loss, test_loss):
    fig = go.Figure()
    trace = go.Scatter(x=np.arange(len(train_loss)), y=train_loss, name="train")
    fig.add_trace(trace)
    trace = go.Scatter(x=np.arange(len(test_loss)), y=test_loss, name="test")
    fig.add_trace(trace)
    fig.update_xaxes(range=[0, None], title="Iteration")
    fig.update_yaxes(range=[0, None], title="Loss")
    return fig


