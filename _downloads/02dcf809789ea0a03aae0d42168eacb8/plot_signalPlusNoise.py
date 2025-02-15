"""
Generation of a signal plus noise time series
=============================================
"""

#%%
# Import requirements
# -------------------

import os
import numpy as np
import plotly.graph_objects as go
import plotly.subplots

#%%
# Define variables
# ----------------

srate = 1
T = 200 # sec
sigma_low = 1.0
sigma_high = 25.0
A = 2.0
omega = 1.0/50
phi = 2*np.pi*15/50

#%%
# Create white noise
# ------------------
#

time = np.arange(0, T, 1.0/srate)
N = len(time)
w_low = np.random.normal(loc=0, scale=sigma_low, size=N)
w_high = np.random.normal(loc=0, scale=sigma_high, size=N)

#%%
# Create signal plus noise time series
# ------------------------------------
#

signal = A*np.cos(2*np.pi*omega*time+phi)
signalInNoise_low = signal + w_low
signalInNoise_high = signal + w_high

#%%
# Plot signal plus noise time series
# ----------------------------------
#

fig = plotly.subplots.make_subplots(
    rows=3, cols=1,
    subplot_titles=(r'$2\cos(2\pi t + 0.6\pi)$',
                    r'$2\cos(2\pi t + 0.6\pi)+N(0,1)$',
                    r'$2\cos(2\pi t + 0.6\pi)+N(0,25)$'))

trace = go.Scatter(x=time, y=signal, mode="lines+markers",
                   line=dict(color="black"), showlegend=False)
fig.add_trace(trace, row=1, col=1)
trace = go.Scatter(x=time, y=signalInNoise_low, mode="lines+markers",
                   line=dict(color="black"), showlegend=False)
fig.add_trace(trace, row=2, col=1)
trace = go.Scatter(x=time, y=signalInNoise_high, mode="lines+markers",
                   line=dict(color="black"), showlegend=False)
fig.add_trace(trace, row=3, col=1)
fig.update_xaxes(title_text="Time (sec)", row=3, col=1)

if not os.path.exists("figures"):
    os.mkdir("figures")

fig.write_html("figures/signalPlusNoise.html")
fig.write_image("figures/signalPlusNoise.png")

fig
