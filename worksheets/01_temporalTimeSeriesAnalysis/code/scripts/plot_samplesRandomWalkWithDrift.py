"""
Random walk with drift: plot 100 samples, mean and 95% confidence interval
==========================================================================
"""

#%%
# Import requirements
# -------------------

import os
import numpy as np
import plotly.graph_objects as go

#%%
# Define variables
# ----------------

srate = 1
T = 100
delta = 1.0
n_time_series_to_plot = 100
sigma = 2.0
cb_width = 5

#%%
# Create white noise
# ------------------
#

time = np.arange(0, T, 1.0/srate)
w = np.random.normal(loc=0, scale=sigma,
                     size=(n_time_series_to_plot, len(time)))

#%%
# Create time series
# ------------------
#
x = np.empty([n_time_series_to_plot, len(w)], dtype=np.double)
x[0] = ... # complete this line
for i in range(1, len(w)):
    x[:, i] = ... # complete this line

#%%
# Calculate analytical mean, variance and 95% confidence band
# -----------------------------------------------------------
#

mean = delta * time
var = ... # complete this line
cb_up = mean + 1.96 * np.sqrt(var)
cb_down = mean - 1.96 * np.sqrt(var)


#%%
# Plot time series with meand and 95% confidence band
# ---------------------------------------------------
#

fig = go.Figure()
for j in range(n_time_series_to_plot):
    trace = go.Scatter(x=time, y=x[j, :], mode="lines+markers",
                       showlegend=False)
    fig.add_trace(trace)
trace = go.Scatter(x=time, y=mean,
                   line=dict(color="black", width=cb_width),
                   mode="lines", showlegend=False)
fig.add_trace(trace)
trace = go.Scatter(x=time, y=cb_down,
                   line=dict(color="black", dash="dash", width=cb_width),
                   mode="lines", showlegend=False)
fig.add_trace(trace)
trace = go.Scatter(x=time, y=cb_up,
                   line=dict(color="black", dash="dash", width=cb_width),
                   mode="lines", showlegend=False)
fig.add_trace(trace)
fig.update_layout(xaxis=dict(title="Time (sec)"), yaxis=dict(title="x"))

if not os.path.exists("figures"):
    os.mkdir("figures")

fig.write_html("figures/randomNoiseWithDriftSamples.html")
fig.write_image("figures/randomNoiseWithDriftSamples.png")

fig
