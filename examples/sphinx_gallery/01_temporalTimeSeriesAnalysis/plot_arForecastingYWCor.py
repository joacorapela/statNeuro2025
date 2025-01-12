
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


#%%
# Define auxiliary functions
# --------------------------

import os
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

N = 10000
M = 500
max_h = 50
sigma_e = 5.0
# ar_coefs = [5.0/6, -1.0/6]
ar_coefs = [5.0/6, -1.0/6, 0.5/6, -0.25/6, 0.5/6, -0.1/6, 0.05/6]

p = len(ar_coefs)

#%%
# Simulate AR(p) time series
# --------------------------

N = 10000
w = np.random.normal(scale=sigma_e, size=N)
x = np.empty(N, dtype=np.double)
x[0] = w[0]
x[1] = w[1]
for i in range(2, N):
    x[i] = 0
    for j in range(p):
        x[i] += ar_coefs[j] * x[i-(1+j)]
    x[i] += w[i]

#%%
# Compute correlations using the Yule-Walker equations
# ----------------------------------------------------

mu_hat = x.mean()
var_hat = x.var()
cor = np.empty(M+max_h, dtype=np.double)
cor[0] = 1.0
for h in range(1, p):
    cor[h] = np.mean((x[h:]-mu_hat)*(x[:-h]-mu_hat))/var_hat
for h in range(p, M+max_h):
    cor[h] = 0
    for i in range(p):
        cor[h] += ar_coefs[i] * cor[h-(i+1)]
cov = cor * var_hat

#%%
# Compute forecasts
# -----------------

Gamma_m = buildGamma_m(cov, M)
forecasts_means = np.empty(max_h, dtype=np.double)
forecasts_vars = np.empty(max_h, dtype=np.double)
xMinusMuR = (x-mu_hat)[::-1][:M]
for h in range(1, max_h+1):
    Gamma_h = buildGamma_h(cov, M, h)
    a_m = np.linalg.solve(Gamma_m, Gamma_h)
    forecasts_means[h-1] = mu_hat + np.inner(a_m, xMinusMuR)
    forecasts_vars[h-1]  = var_hat - np.inner(a_m, cov[h:(h+M)])

#%%
# Plot time series and forecast
# -----------------------------

Gamma_m = buildGamma_m(cov, M)
fig = go.Figure()
trace_x = go.Scatter(x=np.arange(N-M, N), y=x[-M:],
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
