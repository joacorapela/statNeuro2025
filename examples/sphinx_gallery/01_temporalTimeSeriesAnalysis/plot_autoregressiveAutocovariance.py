"""
AR(1) analytical and estimated autocovariance
=============================================
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
T = 10000
sigma = 1.0
phi = -0.9
lags = np.arange(20)

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
for i in range(1, len(w)):
    ar[i] = phi * ar[i-1] + w[i]

#%%
# Estimate autocovariance
# -----------------------
#

estAcov = np.zeros(len(lags))
anaAcov = np.zeros(len(lags))
arMu = np.mean(ar)
for h in lags:
    if h > 0:
        xs = ar[h:] - arMu
        xt = ar[:-h] - arMu
    elif h == 0:
        xs = ar - arMu
        xt = ar - arMu
    estAcov[h] = np.mean(xs * xt)
    anaAcov[h] = phi**h * sigma**2 / (1 - phi**2)

#%%
# Plot autoregressive time series, true and estimated autocovariance
# ------------------------------------------------------------------
#

fig = go.Figure()
trace = go.Scatter(x=time, y=ar, mode="lines+markers")
fig.add_trace(trace)
fig.update_layout(xaxis=dict(title="Time (sec)"), yaxis=dict(title="x"))

if not os.path.exists("figures"):
    os.mkdir("figures")

fig.write_html(f"figures/autoregressiveSamplesN{T}.html")
fig.write_image(f"figures/autoregressiveSamplesN{T}.png")

fig = go.Figure()
trace = go.Scatter(x=lags, y=anaAcov, mode="lines+markers", name="analytical")
fig.add_trace(trace)
trace = go.Scatter(x=lags, y=estAcov, mode="lines+markers", name="estimated")
fig.add_trace(trace)
fig.update_layout(title=f"N={T}", xaxis=dict(title="Lag (samples)"),
                  yaxis=dict(title="Autocovariance"))

if not os.path.exists("figures"):
    os.mkdir("figures")

fig.write_html(f"figures/autoregressiveAutoCovN{T}.html")
fig.write_image(f"figures/autoregressiveAutoCovN{T}.png")

fig
