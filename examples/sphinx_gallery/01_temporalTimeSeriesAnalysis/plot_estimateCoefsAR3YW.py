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

#%%
# Define variables
# ----------------

srate = 1
T = 1000
# T = 50
phi1 = .3
phi2 = .2
phi3 = .1
sigma = 2.0
lags_samples = np.arange(4)

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

x = np.empty(len(w), dtype=np.double)
i = 0
x[i] = w[i]
i = 1
x[i] = phi1 * x[i-1] + w[i]
i = 2
x[i] = phi2 * x[i-2] + phi1 * x[i-1] + w[i]
for i in range(3, len(w)):
    x[i] = phi3 * x[i-3] + phi2 * x[i-2] + phi1 * x[i-1] + w[i]

#%%
# Estimate autocovariance
# -----------------------
#

acov = np.zeros(len(lags_samples))
xMu = np.mean(x)
for h in lags_samples:
    if h > 0:
        xs = x[h:] - xMu
        xt = x[:-h] - xMu
    elif h == 0:
        xs = x - xMu
        xt = x - xMu
    acov[h] = np.mean(xs * xt)

#%%
# Estimate AR(3) coefficients using the Yule-Walker equations
# -----------------------------------------------------------
#

gamma = np.array(acov[1:])
Gamma = np.array([[acov[0], acov[1], acov[2]],
                  [acov[1], acov[0], acov[1]],
                  [acov[2], acov[1], acov[0]]]
                 )
phiHat = np.linalg.solve(Gamma, gamma)
sigma2Hat = gamma[0] - np.dot(gamma, phiHat) 

covPhiEstimates = sigma2Hat/N * np.linalg.inv(Gamma)

#%%
# Plot tru and estimate coefs with 95% confidence intervals
# ---------------------------------------------------------
#

fig = go.Figure()
x = ["1", "2", "3"]
y = phiHat.tolist()
error_y = 1.96 * np.sqrt(np.diag(covPhiEstimates))
trace = go.Scatter(x=x, y=y,
                   mode="markers",
                   error_y=dict(type="data",
                                array=error_y,
                                visible=True),
                   name="estimated",
                   )
fig.add_trace(trace)

y = [phi1, phi2, phi3]
trace = go.Scatter(x=x, y=y,
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

fig.write_html(f"figures/coefficientsAR3N{T*srate}.html")
fig.write_image(f"figures/coefficientsAR3N{T*srate}.png")

fig

breakpoint()
