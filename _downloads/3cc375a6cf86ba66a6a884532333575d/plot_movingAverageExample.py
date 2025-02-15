"""
Generation of a moving average time series
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
# -----------------

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
# Create moving average time series
# ---------------------------------
#
ma = np.empty(len(w), dtype=np.double)
ma[0] = (w[0] + w[1]) / 3
for i in range(1, len(w)-1):
    ma[i] = (w[i-1] + w[i] + w[i+1]) / 3
ma[-1] = (w[-2] + w[-1]) / 3

#%%
# Plot moving average time series
# -------------------------------
#

fig = go.Figure()
trace = go.Scatter(x=time, y=ma, mode="lines+markers")
fig.add_trace(trace)
fig.update_layout(xaxis=dict(title="Time (sec)"), yaxis=dict(title="x"))

if not os.path.exists("figures"):
    os.mkdir("figures")

fig.write_html("figures/movingAverage.html")
fig.write_image("figures/movingAverage.png")

fig
