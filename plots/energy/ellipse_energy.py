import numpy as np
import math
import matplotlib.pyplot as plt

font = {
    "family": "serif",
    "math_fontfamily": "cm",
    "color": "black",
    "weight": "semibold",
    "size": 13,
}

A  = 1.00  #  [m]
freq = 3.0 # [Hz]
wu = 2 * math.pi * freq # [rad/s]
t = np.linspace(1, 1/freq, 256)
et = A * np.sin(wu * t)
det = wu * A * np.cos(wu * t)

stiffness = 100.0
wn = 2 * math.pi * 10
zeta = 1.0

fint = (1 - (wu/wn)**2) * et + 2*zeta/wn * det
Energy = stiffness/2 * (et**2 + det**2/wn**2)
Power = fint*det

# 3D plot with projections:
ax = plt.figure().add_subplot(
    projection="3d", xticklabels=[], yticklabels=[], zticklabels=[]
)
# Set the orthographic projection:
scale_factor = 0.025
ax.set_proj_type("ortho")
ax.plot(et, det, fint, color="k")
ax.plot(et, det, scale_factor * Energy, color="r")
ax.plot(et, det, scale_factor * Power, color="b")

ax.set_ylabel("$\dot{e}_x$", fontdict=font)
ax.set_zlabel("$f_{ext}$", fontdict=font)
ax.set_xlabel("$e_x$", fontdict=font)
plt.show(block=True)
