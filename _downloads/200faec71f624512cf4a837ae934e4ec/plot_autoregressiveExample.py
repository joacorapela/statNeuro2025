"""
Generation of a autoregressive time series
==========================================
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
T = 500
sigma = 1.0

#%%
# Create white noise
# ------------------
#

time = np.arange(0, T, 1.0/srate)
N = len(time)
w = np.random.normal(loc=0, scale=sigma, size=N)

#%%
# Create autoregressive time series
# ---------------------------------
#
ar = np.empty(len(w), dtype=np.double)
ar[0] = w[0]
ar[1] = ar[0] + w[1]
for i in range(2, len(w)):
    ar[i] = ar[i-1] - 0.9 * ar[i-2] + w[i]

#%%
# Plot autoregressive time series
# -------------------------------
#

fig = go.Figure()
trace = go.Scatter(x=time, y=ar, mode="lines+markers")
fig.add_trace(trace)
fig.update_layout(xaxis=dict(title="Time (sec)"), yaxis=dict(title="x"))

if not os.path.exists("figures"):
    os.mkdir("figures")

fig.write_html("figures/autoregressive.html")
fig.write_image("figures/autoregressive.png")

fig
