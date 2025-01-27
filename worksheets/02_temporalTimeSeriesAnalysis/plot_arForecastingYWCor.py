
"""
Forecasting for AR(p) random process
====================================
"""

#%%
# Import requirements
# -------------------

import os
import numpy as np
import plotly.graph_objects as go
import tsAnalysisUtils


#%%
# Define variables
# -----------------

seed = np.nan
N = 10000
m = 500
max_h = 50
sigma = 5.0
# sigma = 5.0
# phi = [-0.9]
phi = [5.0/6, -1.0/6, 0.5/6, -0.25/6, 0.5/6, -0.1/6, 0.05/6]

p = len(phi)

#%%
# Simulate AR(p) time series
# --------------------------

if not np.isnan(seed):
    np.random.seed(seed=seed)
w = np.random.normal(scale=sigma, size=N)
x = tsAnalysisUtils.simulateARp(phi=phi, w=w)

#%%
# Compute correlations using the Yule-Walker equations
# ----------------------------------------------------

acov = np.empty(m+max_h, dtype=np.double)
acov[:p] = tsAnalysisUtils.estimateACov(x=x, lags=np.arange(p))
for h in range(p, m+max_h):
    acov[h] = 0
    for i in range(p):
        acov[h] += phi[i] * acov[h-(i+1)]

#%%
# Compute forecasts
# -----------------

mu = np.mean(x)
forecasts_means, forecasts_vars = tsAnalysisUtils.forecast(x=x, acov=acov,
                                                           mu=mu, m=m,
                                                           max_h=max_h)

#%%
# Plot time series and forecast
# -----------------------------

fig = go.Figure()
trace_x = go.Scatter(x=np.arange(N-m, N), y=x[-m:],
                     name=f"AR({p})")
fig.add_trace(trace_x)

# plot forecast mean
trace_fMean = go.Scatter(x=np.arange(N, N+max_h), y=forecasts_means,
                         name="Forecast")
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

fig.write_html(f"figures/forecastingAR{p}.html")
fig.write_image(f"figures/forecastingAR{p}.png")

fig

