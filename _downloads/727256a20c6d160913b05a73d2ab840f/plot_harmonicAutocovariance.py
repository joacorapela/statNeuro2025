"""
Harmonic process analytical and estimated autocovariance
========================================================
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

min_freq = 1.0 # Hertz
max_freq = 30.0 # Hertz
srate = int(10 * max_freq)
# T = 1.2 # sec
T = 50 # sec
K = 20
# K = 1
max_A = 20.0
min_A = 0.0
lags_samples = np.arange(srate) # samples

#%%
# Set harmonic process constants
# ------------------------------
#

A = np.linspace(start=min_A, stop=max_A, num=K)
f = np.linspace(start=min_freq, stop=max_freq, num=K)
w = 2*np.pi*f
phi = np.random.uniform(low=-np.pi, high=np.pi, size=K)

#%%
# Create samples
# --------------
#

time = np.arange(0, T, 1.0/srate)
x = np.zeros(shape=len(time), dtype=np.double)
for k in range(K):
    x += A[k] * np.cos(w[k] * time + phi[k])

fig = go.Figure()
trace = go.Scatter(x=time, y=x, mode="lines+markers", showlegend=False)
fig.add_trace(trace)
fig.update_layout(title=f"N={T*srate}", xaxis=dict(title="Time (sec)"),
                  yaxis=dict(title="x"))

if not os.path.exists("figures"):
    os.mkdir("figures")

fig.write_html(f"figures/harmonicAutoCovN{T*srate}x.html")
fig.write_image(f"figures/harmonicAutoCovN{T*srate}x.png")

fig

#%%
# Estimate autocovariance
# -----------------------
#

estAcov = np.zeros(len(lags_samples))
anaAcov = np.zeros(len(lags_samples))
xMu = np.mean(x)
for h in lags_samples:
    if h > 0:
        xs = x[h:] - xMu
        xt = x[:-h] - xMu
    elif h == 0:
        xs = x - xMu
        xt = x - xMu
    estAcov[h] = np.mean(xs * xt)
    lag_secs = h/srate
    anaAcov[h] = .5 * np.sum(A**2 * np.cos(w * lag_secs))

#%%
# Plot analytical and estimated autocovariance
# --------------------------------------------
#

lags_secs = lags_samples/srate

fig = go.Figure()
trace = go.Scatter(x=lags_secs, y=anaAcov, mode="lines+markers", name="analytical")
fig.add_trace(trace)
trace = go.Scatter(x=lags_secs, y=estAcov, mode="lines+markers", name="estimated")
fig.add_trace(trace)
fig.update_layout(title=f"N={T*srate}", xaxis=dict(title="Lag (sec)"),
                  yaxis=dict(title="Autocovariance"))

if not os.path.exists("figures"):
    os.mkdir("figures")

fig.write_html(f"figures/harmonicAutoCovN{T*srate}.html")
fig.write_image(f"figures/harmonicAutoCovN{T*srate}.png")

fig

