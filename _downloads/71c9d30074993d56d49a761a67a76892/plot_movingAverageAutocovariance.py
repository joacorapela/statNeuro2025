"""
Moving average analytical and estimated autocovariance
======================================================
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
T = 100
sigma = 1.0
lags = np.arange(20)

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
# Estimate autocovariance
# -----------------------
#

estAcov = np.zeros(len(lags))
anaAcov = np.zeros(len(lags))
maMu = np.mean(ma)
for h in lags:
    if h == 0:
        anaAcov[h] = 3.0 / 9 * sigma**2
        xs = ma - maMu
        xt = ma - maMu
    else:
        xs = ma[h:] - maMu
        xt = ma[:-h] - maMu
        if h == 1:
            anaAcov[h] = 2.0 / 9 * sigma**2
        elif h == 2:
            anaAcov[h] = 1.0 / 9 * sigma**2
    estAcov[h] = np.mean(xs * xt)

#%%
# Plot moving average time series, true and estimated autocovariance
# ------------------------------------------------------------------
#

fig = go.Figure()
trace = go.Scatter(x=time, y=ma, mode="lines+markers")
fig.add_trace(trace)
fig.update_layout(xaxis=dict(title="Time (sec)"), yaxis=dict(title="x"))

if not os.path.exists("figures"):
    os.mkdir("figures")

fig.write_html(f"figures/movingAverageSamplesN{T}.html")
fig.write_image(f"figures/movingAverageSamplesN{T}.png")

fig = go.Figure()
trace = go.Scatter(x=lags, y=anaAcov, mode="lines+markers", name="analytical")
fig.add_trace(trace)
trace = go.Scatter(x=lags, y=estAcov, mode="lines+markers", name="estimated")
fig.add_trace(trace)
fig.update_layout(title=f"N={T}", xaxis=dict(title="Lag (samples)"),
                  yaxis=dict(title="Autocovariance"))

if not os.path.exists("figures"):
    os.mkdir("figures")

fig.write_html(f"figures/movingAverageAutoCovN{T}.html")
fig.write_image(f"figures/movingAverageAutoCovN{T}.png")

fig
