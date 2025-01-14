"""
Generation of a random walk with drift time series
==================================================
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
T = 200 # sec
sigma = 1.0
delta0 = 0.0
delta1 = .2

#%%
# Create white noise
# ------------------
#

time = np.arange(0, T, 1.0/srate)
N = len(time)
w0 = np.random.normal(loc=0, scale=sigma, size=N)
w1 = np.random.normal(loc=0, scale=sigma, size=N)

#%%
# Create random walk with noise time series
# -----------------------------------------
#

rwWithDrift0 = np.empty(N, dtype=np.double)
rwWithDrift1 = np.empty(N, dtype=np.double)
rwWithDrift0[0] = 0.0
rwWithDrift1[0] = 0.0
for i in range(1, N):
    rwWithDrift0[i] = delta0 + rwWithDrift0[i-1] + w0[i]
    rwWithDrift1[i] = delta1 + rwWithDrift1[i-1] + w1[i]

#%%
# Plot random noise with drift time series
# ----------------------------------------
#

fig = go.Figure()
trace = go.Scatter(x=time, y=rwWithDrift0, mode="lines+markers",
                   line=dict(color="blue"), name=r"$\delta=0.0$")
fig.add_trace(trace)
trace = go.Scatter(x=time, y=rwWithDrift1, mode="lines+markers",
                   line=dict(color="black"), name=r"$\delta=0.2$")
fig.add_trace(trace)
trace = go.Scatter(x=[0, T], y=[0, 0], line=dict(color="blue", dash="dot"),
                   mode="lines", showlegend=False)
fig.add_trace(trace)
trace = go.Scatter(x=[0, T], y=[0, delta1*T], line=dict(color="black", dash="dot"),
                   mode="lines", showlegend=False)
fig.add_trace(trace)
fig.update_layout(xaxis=dict(title="Time (sec)"), yaxis=dict(title="x"))

if not os.path.exists("figures"):
    os.mkdir("figures")

fig.write_html("figures/randomWalkWithDrift.html")
fig.write_image("figures/randomWalkWithDrift.png")

fig
