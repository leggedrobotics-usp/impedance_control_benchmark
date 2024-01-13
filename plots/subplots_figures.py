import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["axes.labelpad"] = 0

from plots.transforms_function import *
import math
import numpy as np

rad2deg = 180 / math.pi

joint_impedance_js = np.load("data/js_data_jointimp.npy")
joint_impedance_ts = np.load("data/ts_data_jointimp.npy")
task_impedance_js = np.load("data/js_data_taskimp.npy")
task_impedance_ts = np.load("data/ts_data_taskimp.npy")

task_color = "#2166ac"
joint_color = "#2166ac"
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
time = task_impedance_ts[t_offset:, 0]  # [s]
ee_pos = task_impedance_ts[t_offset:, 1]  # [m]
ee_vel = task_impedance_ts[t_offset:, 2]  # [m/s]
ee_int = task_impedance_ts[t_offset:, 3]  # [N]
ee_acc = task_impedance_ts[t_offset:, 4]  # [m/ss]

plt.figure(figsize=[i3e_2columns_inch, aspect_ratio * i3e_2columns_inch])

ax1 = plt.subplot(2, 4, 1, xticklabels=[], yticklabels=[])
plt.plot(ee_pos, ee_int, color=task_color)
plt.tick_params("y", labelsize=10)
plt.ylabel("$f_{ext}$", fontdict=font)
plt.xlabel("$e_x$", fontdict=font)


plt.subplot(2, 4, 5, xticklabels=[], yticklabels=[])
plt.plot(
    rad2deg * task_impedance_js[t_offset:, 2],
    task_impedance_js[t_offset:, 8],
    color=joint_color,
    linestyle="-.",
)
plt.xlabel("$q_2$", fontdict=font)

# Joint-space Impedance Controller
time = joint_impedance_ts[t_offset:, 0]  # [s]
ee_pos = joint_impedance_ts[t_offset:, 1]  # [m]
ee_vel = joint_impedance_ts[t_offset:, 2]  # [m/s]
ee_int = joint_impedance_ts[t_offset:, 3]  # [N]
ee_acc = joint_impedance_ts[t_offset:, 4]  # [m/ss]

task_color = "#b2182b"
joint_color = "#b2182b"

plt.subplot(2, 4, 2, sharey=ax1, xticklabels=[], yticklabels=[])
plt.plot(ee_pos, ee_int, color=task_color)
plt.tick_params("y", labelsize=10)
plt.xlabel("$e_x$", fontdict=font)

plt.subplot(2, 4, 6, xticklabels=[], yticklabels=[])
plt.plot(
    rad2deg * joint_impedance_js[t_offset:, 2],
    joint_impedance_js[t_offset:, 8],
    color=joint_color,
    linestyle="-.",
)
plt.xlabel("$q_2$", fontdict=font)

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

plt.subplot(2, 4, 3, xticklabels=[], yticklabels=[])
plt.plot(curves[:, 2, 0], curves[:, 2, 2], color=blue1, label=f"Kd = {K[0]}")
plt.plot(curves[:, 1, 0], curves[:, 1, 2], color=blue2, label=f"Kd = {K[1]}")
plt.plot(curves[:, 0, 0], curves[:, 0, 2], color=blue3, label=f"Kd = {K[2]}")
plt.xlabel("$e_x$", fontdict=font)
plt.ylabel("$f_{ext}$", fontdict=font)
plt.legend(loc="upper left", fontsize="x-small")


ax = plt.subplot(
    2, 4, 4, projection="3d", xticklabels=[], yticklabels=[], zticklabels=[]
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

plt.subplot(2, 4, 7, xticklabels=[], yticklabels=[])
plt.plot(curves[:, 2, 1], curves[:, 2, 2], color=blue1, label=f"Kd = {K[0]}")
plt.plot(curves[:, 2, 1], curves[:, 1, 2], color=blue2, label=f"Kd = {K[1]}")
plt.plot(curves[:, 2, 1], curves[:, 0, 2], color=blue3, label=f"Kd = {K[2]}")
plt.xlabel("$\dot{e}_x$", fontdict=font)
plt.ylabel("$f_{ext}$", fontdict=font)

plt.subplot(2, 4, 8, xticklabels=[], yticklabels=[])
plt.plot(curves[:, 2, 0], curves[:, 2, 1], color=blue1, label=f"Kd = {K[0]}")
plt.plot(curves[:, 2, 0], curves[:, 1, 1], color=blue2, label=f"Kd = {K[1]}")
plt.plot(curves[:, 2, 0], curves[:, 0, 1], color=blue3, label=f"Kd = {K[2]}")
plt.xlabel("${e}_x$", fontdict=font)
plt.ylabel("$\dot{e}_x$", fontdict=font)

plt.subplots_adjust(
    left=0.11, bottom=0.10, right=0.9, top=0.90, wspace=0.23, hspace=0.26
)
plt.show(block=True)
