"""
Generation of Gaussian White Noise
==================================
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
T = 500
sigma = 1.0


#%%
# Create white noise
# ------------------
#

time = np.arange(0, T, 1.0/srate)
N = len(time)
w = np.random.normal(loc=0, scale=sigma, size=N)

#%%
# Plot white noise
# ----------------
#

fig = go.Figure()
trace = go.Scatter(x=time, y=w, mode="lines+markers")
fig.add_trace(trace)
fig.update_layout(xaxis=dict(title="Time (sec)"), yaxis=dict(title="x"))

if not os.path.exists("figures"):
    os.mkdir("figures")

fig.write_html("figures/whiteNoise.html")
fig.write_image("figures/whiteNoise.png")

fig
