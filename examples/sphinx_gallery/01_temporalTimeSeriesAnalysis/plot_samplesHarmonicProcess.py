"""
Plotting samples from harmonic process
======================================
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

srate = 1000
T = 1 # sec
K = 20
min_A = 0.0
max_A = 20.0
min_freq = 1.0 # Hertz
max_freq = 50.0 # Hertz
n_time_series_to_plot = 100
cb_width = 5

#%%
# Set harmonic process constants
# ------------------------------
#

A = np.random.uniform(start=min_A, stop=max_A, num=K)
f = np.random.uniform(start=min_freq, stop=max_freq, num=K)
w = 2*np.pi*f
phi = np.random.uniform(low=-np.pi, high=np.pi, size=[K, n_time_series_to_plot])

#%%
# Create samples
# --------------
#

time = np.arange(0, T, 1.0/srate)
x = np.zeros(shape=[n_time_series_to_plot, len(time)], dtype=np.double)
for i in range(n_time_series_to_plot):
    # generate the ith time series
    for k in range(K):
        x[i, :] += A[k] * np.cos(w[k] * time + phi[k, i])

#%%
# Calculate analytical variance and 95% confidence band
# -----------------------------------------------------
#

mean = np.zeros(shape=len(time), dtype=np.double)
var = np.sum(A**2) / 2.0
cb_up = mean + 1.96 * np.sqrt(var)
cb_down = mean - 1.96 * np.sqrt(var)

#%%
# Plot time series with meand and 95% confidence band
# ---------------------------------------------------
#

fig = go.Figure()
for j in range(n_time_series_to_plot):
    trace = go.Scatter(x=time, y=x[j, :], mode="lines+markers",
                       name=f"sample {j}", showlegend=True)
    fig.add_trace(trace)
fig.update_layout(xaxis=dict(title="Time (sec)"), yaxis=dict(title="x"))
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

fig.write_html("figures/harmonicSamples.html")
fig.write_image("figures/harmonicSamples.png")

fig

breakpoint()
