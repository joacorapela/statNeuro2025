"""
Estimation of coefficients of AR(3) model using the Yule-Walker equations
=========================================================================
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
# ----------------

srate = 1
T = 10000
phi = np.array([.3, .2, .1])
sigma = 1.0

p = len(phi)
lags_samples = np.arange(p+1)

#%%
# Create white noise
# ------------------
#

time = np.arange(0, T, 1.0/srate)
N = len(time)
w = np.random.normal(loc=0, scale=sigma, size=N)

#%%
# Create samples
# --------------
#

x = np.zeros(len(w), dtype=np.double)
x[:p] = w[:p]
for i in range(p, len(w)):
    for j in range(p):
        x[i] += phi[j] * x[i-(j+1)]
    x[i] += w[i]

#%%
# Estimate autocovariance
# -----------------------
#

acov = tsAnalysisUtils.estimateACov(x=x, lags=lags_samples)

#%%
# Estimate AR(3) coefficients using the Yule-Walker equations
# -----------------------------------------------------------
#

phiHat, phiCovHat, sigma2Hat = \
    tsAnalysisUtils.estimateCoefsAndNoiseVarARpYW(acov=acov, p=p, N=N)

#%%
# Plot true and estimate coefs with 95% confidence intervals
# ----------------------------------------------------------
#

fig = go.Figure()
x = ["{:d}".format(int(lag)) for lag in range(1, p+1)]
y = phiHat.tolist()
error_y = 1.96 * np.sqrt(np.diag(phiCovHat))
trace = go.Scatter(x=x, y=y,
                   mode="markers",
                   error_y=dict(type="data",
                                array=error_y,
                                visible=True),
                   name="estimated",
                   )
fig.add_trace(trace)

trace = go.Scatter(x=x, y=phi,
                   mode="markers",
                   name="analytical",
                   )
fig.add_trace(trace)
title = r'$\sigma^2={:.2f}, \hat\sigma^2={:.2f}$'.format(sigma**2, sigma2Hat)
fig.update_layout(title=title,
                  xaxis=dict(title="i"),
                  yaxis=dict(title=r"$\phi_i$"))

if not os.path.exists("figures"):
    os.mkdir("figures")

fig.write_html(f"figures/coefficientsAR{p}N{T*srate}.html")
fig.write_image(f"figures/coefficientsAR{p}N{T*srate}.png")

fig

breakpoint()
