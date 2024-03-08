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

pos_err = []
dpos_err = []
f_ext = []

Fo = 50.0  #  [N]
wu = 3.00  # [Hz]

freq_u = wu / (2 * math.pi)
dt = 0.001
steps = int(1 / (dt * freq_u))

# Standard params equivalent to K=1000, wn=250, zeta=1
# consistent with the Pinocchio Simulation, but bad to visualize
params = [
    {"Kd": 1000, "Bd": 8.0, "Md": 0.016, "wu": 3},
    {"Kd": 500,  "Bd": 8.0, "Md": 0.016, "wu": 3},
    {"Kd": 100,  "Bd": 8.0, "Md": 0.016, "wu": 3},
    {"Kd": 1000, "Bd": 4.0, "Md": 0.016, "wu": 3},
    {"Kd": 1000, "Bd": 0.0, "Md": 0.016, "wu": 3},
    {"Kd": 1000, "Bd": 8.0, "Md": 0.160, "wu": 3},
    {"Kd": 1000, "Bd": 8.0, "Md": 1.600, "wu": 3},
]

# Better to visualize
params = [
    {"Kd": 400, "Bd": 40, "Md": 4, "wu": 3},
    {"Kd": 200, "Bd": 40, "Md": 4, "wu": 3},
    {"Kd": 100, "Bd": 40, "Md": 4, "wu": 3},
    {"Kd": 400, "Bd": 20, "Md": 4, "wu": 3},
    {"Kd": 400, "Bd": 4,  "Md": 4, "wu": 3},
    {"Kd": 400, "Bd": 40, "Md": 44.444, "wu": 3},
    {"Kd": 400, "Bd": 40, "Md": 80, "wu": 3},
]

curves = np.zeros(
    (steps, len(params), 3)
)  # 1st dim: time, 2nd dim: params, 3rd dim: {x,y,z}

# Generate Curves Data:
for j in range(len(params)):
    for k in range(steps):
        Tx, Ty, Tz = transforms(params[j])
        curves[k, j, :] = (
            Tx
            @ Ty
            @ Tz
            @ np.array([Fo * math.cos(wu * k * dt), Fo * math.sin(wu * k * dt), 0]).T
        )

check_plot = False
threeD_plot = True

# Test Plot (Plot x(t) vs F(t) for K=100, K=200, K=300)
if check_plot:
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

if threeD_plot:
    # 3D plot with projections:
    ax = plt.figure().add_subplot(
        projection="3d", xticklabels=[], yticklabels=[], zticklabels=[]
    )
    # Set the orthographic projection:
    ax.set_proj_type("ortho")
    x_off = np.max(curves[:, 0, 0]) * 1.05
    y_off = np.max(curves[:, 0, 1]) * 1.05
    z_off = np.min(curves[:, 0, 2]) * 1.05
    ax.plot(curves[:, 0, 0], curves[:, 0, 1], curves[:, 0, 2], color="k")
    ax.plot(
        curves[:, 0, 0],
        curves[:, 0, 1],
        zs=z_off,
        zdir="z",
        color="royalblue",
        linestyle="--",
        alpha=0.90,
    )
    ax.plot(
        curves[:, 0, 0],
        curves[:, 0, 2],
        zs=y_off,
        zdir="y",
        color="palegreen",
        linestyle="--",
        alpha=0.99,
    )
    ax.plot(
        curves[:, 0, 1],
        curves[:, 0, 2],
        zs=x_off,
        zdir="x",
        color="forestgreen",
        linestyle="--",
        alpha=0.90,
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
    plt.show(block=False)

# Access the params easily for the plot legends
K = [params[2]["Kd"], params[1]["Kd"], params[0]["Kd"]]
D = [params[4]["Bd"], params[3]["Bd"], params[0]["Bd"]]
M = [params[0]["Md"], params[5]["Md"], params[6]["Md"]]

# Gradient colors: https://colorbrewer2.org/#type=sequential&scheme=Blues&n=7
blue1 = "#6baed6"
blue2 = "#2171b5"
blue3 = "#084594"

green1 = "#74c476"
green2 = "#238b45"
green3 = "#005a32"

purple1 = "#9e9ac8"
purple2 = "#6a51a3"
purple3 = "#4a1486"

plt.figure(figsize=[11, 4], layout='constrained')
plt.subplot(131, xticklabels=[], yticklabels=[])
plt.plot(curves[:, 2, 0], curves[:, 2, 2], color=blue1, label=f"Kd={K[0]}")
plt.plot(curves[:, 1, 0], curves[:, 1, 2], color=blue2, label=f"Kd={K[1]}")
plt.plot(curves[:, 0, 0], curves[:, 0, 2], color=blue3, label=f"Kd={K[2]}")
plt.xlabel("$e_x$", fontdict=font)
plt.ylabel("$f_{ext}$", fontdict=font)
plt.legend(loc="upper left", fontsize="small", borderpad=0.1)

plt.subplot(132, xticklabels=[], yticklabels=[])
plt.plot(curves[:, 4, 0], curves[:, 4, 2], color=green1, label=f"Bd={D[0]:.2f}")
plt.plot(curves[:, 3, 0], curves[:, 3, 2], color=green2, label=f"Bd={D[1]:.1f}")
plt.plot(curves[:, 0, 0], curves[:, 0, 2], color=green3, label=f"Bd={D[2]:.1f}")
plt.xlabel("$e_x$", fontdict=font)
plt.ylabel("$f_{ext}$", fontdict=font)
plt.legend(loc="upper left", fontsize="small", borderpad=0.1)

plt.subplot(133, xticklabels=[], yticklabels=[])
plt.plot(curves[:, 0, 0], curves[:, 0, 2], color=purple1, label=f"Md={M[0]:.2f}")
plt.plot(curves[:, 5, 0], curves[:, 5, 2], color=purple2, label=f"Md={M[1]:.1f}")
plt.plot(curves[:, 6, 0], curves[:, 6, 2], color=purple3, label=f"Md={M[2]:.1f}")
plt.xlabel("$e_x$", fontdict=font)
plt.ylabel("$f_{ext}$", fontdict=font)
plt.legend(loc="upper center", fontsize="small", ncols=3, bbox_to_anchor=(0.5, 1.08))

#plt.subplots_adjust(
#    left=0.07, bottom=0.07, right=0.97, top=0.93, wspace=0.32, hspace=0.35
#)
plt.show(block=True)
