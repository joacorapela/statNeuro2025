
"""
Forecasting for AR(1) random process
====================================
"""

#%%
# Import requirements
# -------------------

import os
import numpy as np
import plotly.graph_objects as go


#%%
# Define auxiliary functions
# --------------------------

def buildGamma_m(cov, M):
    Gamma_m = np.empty((M, M), dtype=np.double)
    for i in range(M):
        Gamma_m[i, i] = cov[0]
        if i < M-1:
            for j in range(i+1, M):
                Gamma_m[i, j] = cov[j-i]
                Gamma_m[j, i] = Gamma_m[i, j]
    return Gamma_m


def buildGamma_h(cov, M, h):
    Gamma_h = np.empty(M, dtype=np.double)
    for i in range(M):
        Gamma_h[i] = cov[h + i]
    return Gamma_h

#%%
# Define variables
# -----------------

N = 1000
M = 500
max_h = 50
sigma = 1.0
phi = -5.0/6

#%%
# Simulate AR(1) time series
# --------------------------

N = 10000
w = np.random.normal(scale=sigma, size=N)
x = np.empty(N, dtype=np.double)
x[0] = w[0]
for i in range(1, N):
    x[i] = phi * x[i-1] + w[i]

#%%
# Compute covariance
# ------------------

cov = np.empty(M+max_h, dtype=np.double)
var = sigma**2 / (1-phi**2)
for h in range(M+max_h):
    cov[h] = var * phi**h

#%%
# Compute forecasts
# -----------------

mu = 0
Gamma_m = buildGamma_m(cov, M)
forecasts_means = np.empty(max_h, dtype=np.double)
forecasts_vars = np.empty(max_h, dtype=np.double)
xR = x[::-1][:M]
for h in range(1, max_h+1):
    Gamma_h = buildGamma_h(cov, M, h)
    a_m = np.linalg.solve(Gamma_m, Gamma_h)
    forecasts_means[h-1] = mu + np.inner(a_m, xR)
    forecasts_vars[h-1]  = var - np.inner(a_m, cov[h:(h+M)])

#%%
# Plot time series and forecast
# -----------------------------

fig = go.Figure()
trace_x = go.Scatter(x=np.arange(N-M, N), y=x[-M:], mode="lines+markers", name="AR({1})")
fig.add_trace(trace_x)

# plot forecast mean
trace_fMean = go.Scatter(x=np.arange(N, N+max_h), y=forecasts_means,
                         mode="lines+markers", name="Forecast")
fig.add_trace(trace_fMean)

# plot forecast 95% CI
indices = np.arange(N, N+max_h).tolist()
error_f = 1.96 * np.sqrt(forecasts_vars)
y_upper = (forecasts_means + error_f).tolist()
y_lower = (forecasts_means - error_f).tolist()
trace_f95CI = go.Scatter(x=indices+indices[::-1],
                         y=y_upper+y_lower[::-1],
                         fill="toself",
                         fillcolor="rgba(0,100,80,0.2)",
                         line=dict(color="rgba(255,255,255,0.0)"),
                         hoverinfo="skip",
                         showlegend=False)
fig.add_trace(trace_f95CI)

fig.update_layout(xaxis_title="Sample Index", yaxis_title="Process Value")

if not os.path.exists("figures"):
    os.mkdir("figures")

fig.write_html("figures/forecastingAR1.html")
fig.write_image("figures/forecastingAR1.png")

fig
