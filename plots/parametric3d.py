import numpy as np
import time as tm
import math
import matplotlib.pyplot as plt
import platform
from transforms_function import *

if platform.system() == 'Windows':
    log_path = "impedance_control_benchmark/data"
if platform.system() == 'Linux':
    log_path = "data"
task_impedance_ts = np.load(log_path + "/ts_data_taskimp.npy")[700:, :]

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
    {"Kd": 1000, "Dd": 8, "Md": 0.000, "wu": wu},
    {"Kd": 1000, "Dd": 8, "Md": 44.00, "wu": wu},
]

curves = np.zeros(
    (steps, len(parameters_set), 3)
)  # 1st dim: time, 2nd dim: parameters_set, 3rd dim: {x,y,z}

# Generate Curves:
for j, param in enumerate(parameters_set):
    for k in range(steps):
        Tx, Ty, Tz = transforms(param)
        curves[k, j, :] = (
            Tx
            @ Ty
            @ Tz
            @ np.array([Amp * math.cos(wu * k * dt), Amp * math.sin(wu * k * dt), 0]).T
        )

# 3D plot with projections:
ax = plt.figure().add_subplot(
    projection="3d", xticklabels=[], yticklabels=[], zticklabels=[]
)
ax.set_proj_type("ortho")

curve_color = ["green", "darkviolet"]
projections_offset = 1.25
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
    pvalue = pset["Md"]
    ax.plot(curves[:, c, 0], curves[:, c, 1], curves[:, c, 2],
            color=curve_color[c], label=f"Md={pvalue:.3f}")

    #ax.plot(curves[:, c, 0], curves[:, c, 1],
    #        zs=z_off, zdir="z", color=curve_color[c+2],
    #        linestyle="--",
    #)
    ax.plot(curves[:, c, 0], curves[:, c, 2],
            zs=y_off, zdir="y", color=curve_color[c],
            linestyle="--", alpha=0.7,
    )
    ax.plot(curves[:, c, 1], curves[:, c, 2],
            zs=x_off, zdir="x", color=curve_color[c],
            linestyle="--", alpha=0.7,
    )

# Log from Pinocchio simulation
ax.plot(task_impedance_ts[:, 1],
        task_impedance_ts[:, 2],
        task_impedance_ts[:, 3],
        color="k", alpha=0.0)

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
           ncols=3, bbox_to_anchor=(0.5, 1.00)
)
plt.show(block=True)
