import numpy as np
import math
import matplotlib.pyplot as plt
from transforms_function import *

task_impedance_ts = np.load("data/ts_data_taskimp.npy")[700:, :]

# font dictionary to standardize
font = {
    "family": "serif",
    "math_fontfamily": "cm",
    "color": "black",
    "weight": "semibold",
    "size": 13,
}

Amp = 9.80665 * 5 / 1000 # [m]
wu  = 3.00  # [Hz]
dt  = 0.01
steps = math.ceil(2 * math.pi / (dt * wu))

simulation_params = {"Kd": 1000, "Dd": 8, "Md": 0.016, "wu": wu}
parameters_set = [
    {"Kd": 0.00, "Dd": 0.0, "Md": 0, "wu": wu},
    {"Kd": 1000, "Dd": 0.0, "Md": 0, "wu": wu},
    {"Kd": 1000, "Dd": 300, "Md": 0, "wu": wu},
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
            np.array([math.cos(wu * k * dt), math.sin(wu * k * dt)]).T
        )

# 3D plot with projections:
ax = plt.figure().add_subplot(
    projection="3d", xticklabels=[], yticklabels=[], zticklabels=[]
)
ax.set_proj_type("ortho")

curve_color = ["silver" ,"green", "darkviolet"]
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

# Dashed lines to aid on the angles visual representation
ax.plot([0, Amp], [0, 0], [0, 0], linestyle="--", color="k")
ax.plot([0, Amp], [0, 0], [0, 1000 * Amp], linestyle="--", color="k")
ax.plot([0, 0], [0, -wu * Amp], [0, 0], linestyle="--", color="k")
Tx, Ty, Tz = transforms(parameters_set[2])
yaxis_trans = Tx @ Ty @ Tz @ np.array([0, -Amp])
ax.plot([0, yaxis_trans[0]], [0, yaxis_trans[1]],
        [0, yaxis_trans[2]], linestyle="--", color="k")

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
plt.show(block=True)
