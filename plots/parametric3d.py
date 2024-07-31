# Impedance Control Benchmark
# Copyright (C) 2024, leggedrobotics-usp
# Leonardo F. dos Santos, Cícero L. A. Zanette, and Elisa G. Vergamini
#
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License 3.0,
# or later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import numpy as np
import math
import matplotlib.pyplot as plt
from parametric_analysis.transforms_function import *

task_impedance_ts = np.load("data/ts_data_taskimp.npy")[700:, :]

# font dictionary to standardize
font = {
    "family": "serif",
    "math_fontfamily": "cm",
    "color": "black",
    "weight": "normal",
    "size": 14,
}

Amp = 9.80665 * 5 / 1000 # [m]
wu  = 3.00  # [Hz]
dt  = 0.01
steps = math.ceil(2 * math.pi / (dt * wu))

simulation_params = {"Kd": 1000, "Dd": 8, "Md": 0.016, "wu": wu}
parameters_set = [
    {"Kd": 12, "Dd": 0, "Md": 0, "wu": wu},
    {"Kd": 12, "Dd": 2, "Md": 0, "wu": wu},
    {"Kd":  0, "Dd": 2, "Md": 0, "wu": wu},
    {"Kd": 12, "Dd": 2, "Md": 0, "wu": wu},
]

# 1st dim: time, 2nd dim: parameters_set, 3rd dim: {x,y,z}
curves = np.zeros(
    (steps, len(parameters_set), 3)
)
# Generate Curves:
for j, param in enumerate(parameters_set):
    for k in range(steps):
        Tx, Ty, Tz = transforms(param)
        T = Tx @ Ty @ Tz
        curves[k, j, :] = (Amp * T @
            np.array([math.cos(wu * k * dt), math.sin(wu * k * dt), 0]).T
        )

# 2D Plots
ax = plt.subplot2grid((1, 2), (0, 0), colspan=1)
ax.plot(curves[:, 0, 0], curves[:, 0, 2], label="$k_d$ = 12, $d_d$ = 0")
ax.plot(curves[:, 1, 0], curves[:, 1, 2], label="$k_d$ = 12, $d_d$ = 2")
ax.set_xlabel("$e [m]$", fontdict=font, va='center')
ax.set_ylabel("$f_{int} [N]$", fontdict=font, va='center')
ax.tick_params(axis='both', direction='in')
plt.legend(loc="upper center", fontsize="small", ncols=1, bbox_to_anchor=(0.3, 1.0))

ax2 = plt.subplot2grid((1, 2), (0, 1), colspan=1, yticklabels=[])
ax2.plot(curves[:, 2, 1], curves[:, 2, 2], label="$k_d$ = 0, $d_d$ = 2")
ax2.plot(curves[:, 3, 1], curves[:, 3, 2], label="$k_d$ = 12, $d_d$ = 2")
ax2.set_xlabel("$\dot{e} [m/s]$ ", fontdict=font, va='center')
ax2.tick_params(axis='both', direction='in')
#ax2.set_ylabel("$f_{int}$", fontdict=font, va='center')
plt.legend(loc="upper center", fontsize="small", ncols=1, bbox_to_anchor=(0.3, 1.0))

plt.tight_layout()
plt.show(block=False)

# 3D plot with projections:
ax = plt.figure().add_subplot(
    projection="3d", xticklabels=[], yticklabels=[], zticklabels=[]
)
ax.set_proj_type("ortho")

curve_color = ["silver" ,"green", "darkviolet", "blue"]
projections_offset = 1.45
projections_alpha  = 0.0
plot_type = 'surface'

x_off = np.max(curves[:, 0, 0]) * projections_offset
y_off = np.max(curves[:, 0, 1]) * projections_offset
z_off = np.min(curves[:, 0, 2]) * projections_offset

for c in range(len(parameters_set)):
    x_off_new = np.max(curves[:, c, 0]) * projections_offset
    y_off_new = np.max(curves[:, c, 1]) * projections_offset
    z_off_new = np.min(curves[:, c, 2]) * projections_offset
    if x_off_new > x_off:
        x_off = x_off_new
    if y_off_new > y_off:
        y_off = y_off_new
    if z_off_new < z_off:
        z_off = z_off_new

for c, pset in enumerate(parameters_set):
    k_val, d_val = pset["Kd"], pset["Dd"]
    label_string = (
        "$K_d$ = " + f"{k_val}, " +
        "$D_d$ = " + f"{d_val}"
    )
    ax.plot(curves[:, c, 0], curves[:, c, 1],
            curves[:, c, 2], color=curve_color[c],
            label=label_string)

    if plot_type == "contour":
      ax.plot(curves[:, c, 0], curves[:, c, 2],
        zs=y_off, zdir="y", color=curve_color[c],
        linestyle="--", alpha=projections_alpha,
      )
      ax.plot(curves[:, c, 1], curves[:, c, 2],
        zs=x_off, zdir="x", color=curve_color[c],
        linestyle="--", alpha=projections_alpha,
      )
    if plot_type == "surface":
        ax.plot_trisurf(curves[:, c, 0], curves[:, c, 1],
            curves[:, c, 2], color=curve_color[c],
            linewidth=0.3, antialiased=True, alpha=0.5
        )


# Log from Pinocchio simulation
ax.plot(task_impedance_ts[:, 1],
        task_impedance_ts[:, 2],
        task_impedance_ts[:, 3],
        color="k", linestyle=":",
        alpha=0.0)

ax.xaxis.set_rotate_label(False)
ax.yaxis.set_rotate_label(False)
ax.zaxis.set_rotate_label(False)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.set_ylabel("$\dot{e}_x$", fontdict=font)
ax.set_zlabel("$f_{ext}$", fontdict=font)
ax.set_xlabel("$e_x$", fontdict=font)
plt.legend(loc="upper center", fontsize="small",
           ncols=1, bbox_to_anchor=(0.07, 1.00)
)
plt.show(block=False)

ax2 = plt.figure().add_subplot(
    projection="3d", xticklabels=[], yticklabels=[], zticklabels=[]
)
ax2.set_proj_type("ortho")
ax2.set_aspect("equal")

parameters_set = [
    {"Kd": 0, "Dd": 0.0, "Md": 0, "wu": 1.5},
    {"Kd": 1, "Dd": 0.0, "Md": 0, "wu": 1.5},
    {"Kd": 1, "Dd": 0.3, "Md": 0, "wu": 1.5},
]

ax2.plot([0, 1], [0, 0], [0, 0], linestyle="-", color="blue")
ax2.plot([0, 0], [0, 1], [0, 0], linestyle="-", color="blue")
ax2.plot([0, 0], [0, 0], [0, 1], linestyle="-", color="blue")

Tx, Ty, Tz = transforms(parameters_set[0])
Basis = Tx @ Ty @ Tz

ax2.plot([0, Basis[0,0]], [0, Basis[0,1]], [0, Basis[0,2]], linestyle="--", color="grey")
ax2.plot([0, Basis[1,0]], [0, Basis[1,1]], [0, Basis[1,2]], linestyle="--", color="grey")
ax2.plot([0, Basis[2,0]], [0, Basis[2,1]], [0, Basis[2,2]], linestyle="--", color="grey")

Tx, Ty, Tz = transforms(parameters_set[1])
Basis = Tx @ Ty @ Tz

ax2.plot([0, Basis[0,0]], [0, Basis[0,1]], [0, Basis[0,2]], linestyle="--", color="green")
ax2.plot([0, Basis[1,0]], [0, Basis[1,1]], [0, Basis[1,2]], linestyle="--", color="green")
ax2.plot([0, Basis[2,0]], [0, Basis[2,1]], [0, Basis[2,2]], linestyle="--", color="green")

Tx, Ty, Tz = transforms(parameters_set[2])
Basis = Tx @ Ty @ Tz

ax2.plot([0, Basis[0,0]], [0, Basis[0,1]], [0, Basis[0,2]], linestyle="--", color="darkviolet")
ax2.plot([0, Basis[1,0]], [0, Basis[1,1]], [0, Basis[1,2]], linestyle="--", color="darkviolet")
ax2.plot([0, Basis[2,0]], [0, Basis[2,1]], [0, Basis[2,2]], linestyle="--", color="darkviolet")

plt.show(block=True)
