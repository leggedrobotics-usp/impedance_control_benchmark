import numpy as np
import time as tm
import math
import matplotlib.pyplot as plt

from transforms_function import *

# font dictionary to standardize
font = {
    "family": "serif",
    "math_fontfamily": "cm",
    "color": "black",
    "weight": "semibold",
    "size": 13,
}

Fo = 50.0  #  [N]*
wu = 3.00  # [Hz]

freq_u = wu / (2 * math.pi)
dt = 0.001
steps = int(1 / (dt * freq_u))

parameters_set = [
    {"Kd": 400, "Dd": 40, "Md": 0.01, "wu": 3},
    {"Kd": 400, "Dd": 40, "Md": 4, "wu": 3},
    {"Kd": 400, "Dd": 40, "Md": 44.444, "wu": 3},
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
            @ np.array([Fo * math.cos(wu * k * dt), Fo * math.sin(wu * k * dt), 0]).T
        )

CHECK = False

# Test Plot (Plot x(t) vs F(t))
if CHECK:
    plt.plot(
        curves[:, 0, 0],
        curves[:, 0, 2],
        curves[:, 1, 0],
        curves[:, 1, 2],
        curves[:, 2, 0],
        curves[:, 2, 2],
    )
    plt.grid()
    plt.show(block=False)


# 3D plot with projections:
ax = plt.figure().add_subplot(
    projection="3d", xticklabels=[], yticklabels=[], zticklabels=[]
)
ax.set_proj_type("ortho")

curve_color = ["#9e9ac8", "#6a51a3", "#4a1486"]

x_off = np.max(curves[:, 0, 0]) * 1.05
y_off = np.max(curves[:, 0, 1]) * 1.05
z_off = np.min(curves[:, 0, 2]) * 1.05
for c in range(len(parameters_set)):
    x_off_new = np.max(curves[:, c, 0]) * 1.05
    y_off_new = np.max(curves[:, c, 1]) * 1.05
    z_off_new = np.min(curves[:, c, 2]) * 1.05
    if x_off_new > x_off:
        x_off = x_off_new
    if y_off_new > y_off:
        y_off = y_off_new
    if z_off_new < z_off:
        z_off = z_off_new

for c, pset in enumerate(parameters_set):
    pvalue = pset["Md"]
    ax.plot(curves[:, c, 0], curves[:, c, 1], curves[:, c, 2],
            color=curve_color[c], label=f"Md={pvalue:.2f}")

    ax.plot(curves[:, c, 0], curves[:, c, 1],
            zs=z_off, zdir="z", color="royalblue",
            linestyle="--", alpha=0.90,
    )
    ax.plot(curves[:, c, 0], curves[:, c, 2],
            zs=y_off, zdir="y", color="crimson",
            linestyle="--", alpha=0.99,
    )
    ax.plot(curves[:, c, 1], curves[:, c, 2],
            zs=x_off, zdir="x", color="forestgreen",
            linestyle="--", alpha=0.90,
    )

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
           ncols=3, bbox_to_anchor=(0.5, 1.08)
)
plt.show(block=True)
