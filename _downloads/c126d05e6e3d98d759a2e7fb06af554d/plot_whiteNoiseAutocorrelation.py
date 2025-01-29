"""
White noise analytical and estimated autocorrelation, and 95% confidence interval
=================================================================================
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
T = 10000
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
# Estimate autocorrelation
# ------------------------
#

anaAcor = np.zeros(len(lags))
anaAcor[0] = sigma
estAcor = np.zeros(len(lags))
wMu = np.mean(w)
wSTD = np.std(w)
for h in lags:
    if h == 0:
        estAcor[h] = 1.0
    else:
        xs = w[h:] - wMu
        xt = w[:-h] - wMu
        estAcor[h] = np.mean(xs * xt)/wSTD

#%%
# Plot white noise analytical and estimated autocorrelations plus 95% confidence interval
# ---------------------------------------------------------------------------------------
#

fig = go.Figure()
trace = go.Scatter(x=lags, y=anaAcor, mode="lines+markers", name="analytical")
fig.add_trace(trace)
trace = go.Scatter(x=lags, y=estAcor, mode="lines+markers", name="estimated")
fig.add_trace(trace)
fig.add_hline(y=1.96/np.sqrt(N), line=dict(dash="dash"))
fig.add_hline(y=-1.96/np.sqrt(N), line=dict(dash="dash"))
fig.update_layout(title=f"N={T}", xaxis=dict(title="Lag (samples)"),
                  yaxis=dict(title="Autocorrelation"))

if not os.path.exists("figures"):
    os.mkdir("figures")

fig.write_html(f"figures/whiteNoiseAutoCorN{T}.html")
fig.write_image(f"figures/whiteNoiseAutoCorN{T}.png")

fig
