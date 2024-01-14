import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["axes.labelpad"] = 0

import math
import numpy

rad2deg = 180 / math.pi
t_begin = 700  # samples [dt = 0.001 s]

joint_impedance_js = numpy.load("data/js_data_jointimp.npy")[t_begin:, :]
joint_impedance_ts = numpy.load("data/ts_data_jointimp.npy")[t_begin:, :]
task_impedance_js  = numpy.load("data/js_data_taskimp.npy")[t_begin:, :]
task_impedance_ts  = numpy.load("data/ts_data_taskimp.npy")[t_begin:, :]

ts_color = "#2166ad"
js_color = "#b2182c"
font = {
    "family": "serif",
    "math_fontfamily": "cm",
    "color": "black",
    "weight": "semibold",
    "size": 13,
}

subplot_cols = 2
subplot_rows = 3
ieee_width_in = 3.5 + 0.25 + 3.5
aspect_ratio = 9 / 16

# Task-space Impedance Controller
#time = task_impedance_ts[t_offset:, 0]  # [s]
#ee_pos = task_impedance_ts[t_offset:, 1]  # [m]
#ee_vel = task_impedance_ts[t_offset:, 2]  # [m/s]
#ee_int = task_impedance_ts[t_offset:, 3]  # [N]
#ee_acc = task_impedance_ts[t_offset:, 4]  # [m/ss]

plt.figure(figsize=[ieee_width_in, aspect_ratio * ieee_width_in])

ax1 = plt.subplot2grid((subplot_rows, subplot_cols), (1,0), colspan=1, xticklabels=[], yticklabels=[])

ax1.plot(task_impedance_ts[:, 1], task_impedance_ts[:, 3], color=ts_color)
ax1.plot(joint_impedance_ts[:, 1], joint_impedance_ts[:, 3], color=js_color)
ax1.set_xlabel("$e_x$", fontdict=font)
ax1.set_ylabel("$f_{int}$", fontdict=font)

ax2 = plt.subplot2grid((subplot_rows, subplot_cols), (0,0), colspan=1, xticklabels=[], yticklabels=[])
ax2.plot(
    rad2deg * task_impedance_js[:, 2],
    task_impedance_js[:, 8],
    color=ts_color,
    linestyle="-.",
)
ax2.plot(
    rad2deg * joint_impedance_js[:, 2],
    joint_impedance_js[:, 8],
    color=js_color,
    linestyle="-.",
)
ax2.set_ylabel("$tau$", fontdict=font)
ax2.set_xlabel("$q$", fontdict=font)


ax3 = plt.subplot2grid((subplot_rows, subplot_cols), (0,1), rowspan=2, projection="3d", xticklabels=[], yticklabels=[], zticklabels=[])
ax3.set_proj_type("ortho")
ax3.plot(task_impedance_ts[:, 1], task_impedance_ts[:, 2], task_impedance_ts[:, 3], color=ts_color)
ax3.plot(joint_impedance_ts[:, 1], joint_impedance_ts[:, 2], joint_impedance_ts[:, 3], color=js_color)

ax3.set_xlabel("$e_x$", fontdict=font)
ax3.set_ylabel("$\dot{e}_x$", fontdict=font)
ax3.set_zlabel("$f_{int}$", fontdict=font)

ax4 = plt.subplot2grid((subplot_rows, subplot_cols), (2,0), colspan=1, xticklabels=[], yticklabels=[])
ax4.plot(task_impedance_ts[:, 1], task_impedance_ts[:, 2], color=ts_color)
ax4.plot(joint_impedance_ts[:, 1], joint_impedance_ts[:, 2], color=js_color)
ax4.set_ylabel("$\dot{e}_x$", fontdict=font)
ax4.set_xlabel("$e_x$", fontdict=font)

ax5 = plt.subplot2grid((subplot_rows, subplot_cols), (2,1), colspan=1, xticklabels=[], yticklabels=[])
ax5.plot(task_impedance_ts[:, 2], task_impedance_ts[:, 3], color=ts_color)
ax5.plot(joint_impedance_ts[:, 2], joint_impedance_ts[:, 3], color=js_color)
ax5.set_ylabel("$f_{int}$", fontdict=font)
ax5.set_xlabel("$\dot{e}_x$", fontdict=font)

plt.subplots_adjust(
    left=0.11, bottom=0.10, right=0.9, top=0.90, wspace=0.23, hspace=0.26
)
plt.show(block=True)
