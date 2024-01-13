import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["axes.labelpad"] = 0

from transforms_funct import *
from math import pi as pie
import numpy as np

rad2deg = 180 / pie
# Import logs
jsdata_jimp = np.load("data/js_data_jointimp.npy")
tsdata_jimp = np.load("data/ts_data_jointimp.npy")
jsdata_timp = np.load("data/js_data_taskimp.npy")
tsdata_timp = np.load("data/ts_data_taskimp.npy")


ts_color = "#2166ac"
js_color = "#2166ac"
font = {
    "family": "serif",
    "math_fontfamily": "cm",
    "color": "black",
    "weight": "semibold",
    "size": 13,
}

i3e_2columns_inch = 3.5 + 0.25 + 3.5
aspect_ratio = 9 / 16

pos_err = []
dpos_err = []
f_ext = []

Fo = 50.0  #  [N]
wu = 3.00  # [Hz]

freq_u = wu / (2 * math.pi)
dt = 0.001
steps = int(1 / (dt * freq_u))

params = [
    {"Kd": 400, "Bd": 40, "Md": 4, "wu": 3},
    {"Kd": 200, "Bd": 40, "Md": 4, "wu": 3},
    {"Kd": 100, "Bd": 40, "Md": 4, "wu": 3},
    {"Kd": 400, "Bd": 20, "Md": 4, "wu": 3},
    {"Kd": 400, "Bd": 4, "Md": 4, "wu": 3},
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

t_offset = 700  # samples [dt = 0.001 s]
# Task-space Impedance Controller
time = tsdata_timp[t_offset:, 0]  # [s]
ee_pos = tsdata_timp[t_offset:, 1]  # [m]
ee_vel = tsdata_timp[t_offset:, 2]  # [m/s]
ee_int = tsdata_timp[t_offset:, 3]  # [N]
ee_acc = tsdata_timp[t_offset:, 4]  # [m/ss]

plt.figure(figsize=[i3e_2columns_inch, aspect_ratio * i3e_2columns_inch])

ax1 = plt.subplot(2, 9, 1, xticklabels=[], yticklabels=[])
plt.plot(ee_pos, ee_int, color=ts_color)
plt.tick_params("y", labelsize=10)
plt.ylabel("$f_{ext}$", fontdict=font)
plt.xlabel("$e_x$", fontdict=font)

plt.subplot(2, 9, 2, sharey=ax1, xticklabels=[], yticklabels=[])
plt.plot(ee_vel, ee_int, color=ts_color)
plt.xlabel("$\dot{e}_x$", fontdict=font)

plt.subplot(2, 9, 3, sharey=ax1, xticklabels=[], yticklabels=[])
plt.plot(ee_acc, ee_int, color=ts_color)
plt.xlabel("$\ddot{e}_x$", fontdict=font)

plt.subplot(2, 9, 10, xticklabels=[], yticklabels=[])
plt.plot(
    rad2deg * jsdata_timp[t_offset:, 1],
    jsdata_timp[t_offset:, 7],
    color=js_color,
    linestyle="-.",
)
plt.ylabel("$\\tau_i$", fontdict=font)
plt.xlabel("$q_1$", fontdict=font)

plt.subplot(2, 9, 11, xticklabels=[], yticklabels=[])
plt.plot(
    rad2deg * jsdata_timp[t_offset:, 2],
    jsdata_timp[t_offset:, 8],
    color=js_color,
    linestyle="-.",
)
plt.xlabel("$q_2$", fontdict=font)

plt.subplot(2, 9, 12, xticklabels=[], yticklabels=[])
plt.plot(
    rad2deg * jsdata_timp[t_offset:, 3],
    jsdata_timp[t_offset:, 9],
    color=js_color,
    linestyle="-.",
)
plt.xlabel("$q_3$", fontdict=font)

# Joint-space Impedance Controller
time = tsdata_jimp[t_offset:, 0]  # [s]
ee_pos = tsdata_jimp[t_offset:, 1]  # [m]
ee_vel = tsdata_jimp[t_offset:, 2]  # [m/s]
ee_int = tsdata_jimp[t_offset:, 3]  # [N]
ee_acc = tsdata_jimp[t_offset:, 4]  # [m/ss]

ts_color = "#b2182b"
js_color = "#b2182b"

plt.subplot(2, 9, 4, sharey=ax1, xticklabels=[], yticklabels=[])
plt.plot(ee_pos, ee_int, color=ts_color)
plt.tick_params("y", labelsize=10)
plt.xlabel("$e_x$", fontdict=font)

plt.subplot(2, 9, 5, sharey=ax1, xticklabels=[], yticklabels=[])
plt.plot(ee_vel, ee_int, color=ts_color)
plt.xlabel("$\dot{e}_x$", fontdict=font)

plt.subplot(2, 9, 6, sharey=ax1, xticklabels=[], yticklabels=[])
plt.plot(ee_acc, ee_int, color=ts_color)
plt.xlabel("$\ddot{e}_x$", fontdict=font)

plt.subplot(2, 9, 13, xticklabels=[], yticklabels=[])
plt.plot(
    rad2deg * jsdata_jimp[t_offset:, 1],
    jsdata_jimp[t_offset:, 7],
    color=js_color,
    linestyle="-.",
)
plt.xlabel("$q_1$", fontdict=font)

plt.subplot(2, 9, 14, xticklabels=[], yticklabels=[])
plt.plot(
    rad2deg * jsdata_jimp[t_offset:, 2],
    jsdata_jimp[t_offset:, 8],
    color=js_color,
    linestyle="-.",
)
plt.xlabel("$q_2$", fontdict=font)

plt.subplot(2, 9, 15, xticklabels=[], yticklabels=[])
plt.plot(
    rad2deg * jsdata_jimp[t_offset:, 3],
    jsdata_jimp[t_offset:, 9],
    color=js_color,
    linestyle="-.",
)
plt.xlabel("$q_3$", fontdict=font)


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

plt.subplot(2, 9, 7, xticklabels=[], yticklabels=[])
plt.plot(curves[:, 2, 0], curves[:, 2, 2], color=blue1, label=f"Kd = {K[0]}")
plt.plot(curves[:, 1, 0], curves[:, 1, 2], color=blue2, label=f"Kd = {K[1]}")
plt.plot(curves[:, 0, 0], curves[:, 0, 2], color=blue3, label=f"Kd = {K[2]}")
plt.xlabel("$e_x$", fontdict=font)
plt.ylabel("$f_{ext}$", fontdict=font)
plt.legend(loc="upper left", fontsize="x-small")

plt.subplot(2, 9, 8, xticklabels=[], yticklabels=[])
plt.plot(curves[:, 4, 0], curves[:, 4, 2], color=green1, label=f"Bd = {D[0]:.2f}")
plt.plot(curves[:, 3, 0], curves[:, 3, 2], color=green2, label=f"Bd = {D[1]:.1f}")
plt.plot(curves[:, 0, 0], curves[:, 0, 2], color=green3, label=f"Bd = {D[2]:.1f}")
plt.xlabel("$e_x$", fontdict=font)
plt.legend(loc="upper left", fontsize="x-small")

plt.subplot(2, 9, 9, xticklabels=[], yticklabels=[])
plt.plot(curves[:, 0, 0], curves[:, 0, 2], color=purple1, label=f"Md = {M[0]:.2f}")
plt.plot(curves[:, 5, 0], curves[:, 5, 2], color=purple2, label=f"Md = {M[1]:.1f}")
plt.plot(curves[:, 6, 0], curves[:, 6, 2], color=purple3, label=f"Md = {M[2]:.1f}")
plt.xlabel("$e_x$", fontdict=font)
plt.legend(loc="upper left", fontsize="x-small")

ax = plt.subplot(
    2, 9, 16, projection="3d", xticklabels=[], yticklabels=[], zticklabels=[]
)
ax.set_proj_type("ortho")
ax.plot(
    curves[:, 2, 0], curves[:, 2, 1], curves[:, 2, 2], color=blue1, label=f"Kd = {K[0]}"
)
ax.plot(
    curves[:, 1, 0], curves[:, 2, 1], curves[:, 1, 2], color=blue2, label=f"Kd = {K[1]}"
)
ax.plot(
    curves[:, 0, 0], curves[:, 2, 1], curves[:, 0, 2], color=blue3, label=f"Kd = {K[2]}"
)

ax.set_xlabel("$e_x$", fontdict=font)
ax.set_ylabel("$\dot{e}_x$", fontdict=font)
ax.set_zlabel("$f_{ext}$", fontdict=font)

plt.subplot(2, 9, 17, xticklabels=[], yticklabels=[])
plt.plot(curves[:, 2, 1], curves[:, 2, 2], color=blue1, label=f"Kd = {K[0]}")
plt.plot(curves[:, 2, 1], curves[:, 1, 2], color=blue2, label=f"Kd = {K[1]}")
plt.plot(curves[:, 2, 1], curves[:, 0, 2], color=blue3, label=f"Kd = {K[2]}")
plt.xlabel("$\dot{e}_x$", fontdict=font)
plt.ylabel("$f_{ext}$", fontdict=font)

plt.subplot(2, 9, 18, xticklabels=[], yticklabels=[])
plt.plot(curves[:, 2, 0], curves[:, 2, 1], color=blue1, label=f"Kd = {K[0]}")
plt.plot(curves[:, 2, 0], curves[:, 1, 1], color=blue2, label=f"Kd = {K[1]}")
plt.plot(curves[:, 2, 0], curves[:, 0, 1], color=blue3, label=f"Kd = {K[2]}")
plt.xlabel("${e}_x$", fontdict=font)
plt.ylabel("$\dot{e}_x$", fontdict=font)

plt.subplots_adjust(
    left=0.11, bottom=0.10, right=0.9, top=0.90, wspace=0.23, hspace=0.26
)
plt.show(block=True)
