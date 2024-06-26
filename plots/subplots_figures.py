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

import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy
import math

rad2deg = 180 / math.pi
t_begin = 700  # samples [dt = 0.001 s]
joint_impedance_js = numpy.load("data/js_data_jointimp.npy")[t_begin:, :]
joint_impedance_ts = numpy.load("data/ts_data_jointimp.npy")[t_begin:, :]
task_impedance_js  = numpy.load("data/js_data_taskimp.npy")[t_begin:, :]
task_impedance_ts  = numpy.load("data/ts_data_taskimp.npy")[t_begin:, :]

ts_color = colormaps['tab20c'](0)
js_color = colormaps['tab20c'](4)
font = {
    "family": "serif",
    "math_fontfamily": "cm",
    "color": "black",
    "weight": "bold",
    "size": 13,
}

subplot_cols = 2
subplot_rows = 3
ieee_width_in = 7  # + 0.25 + 3.5
aspect_ratio = 9 / 16

scaling = math.e
joint_impedance_ts[:, 1] = scaling * joint_impedance_ts[:, 1]
joint_impedance_ts[:, 2] = scaling * joint_impedance_ts[:, 2]

fig = plt.figure(figsize=[ieee_width_in, aspect_ratio * ieee_width_in])

ax1 = plt.subplot2grid(
    (subplot_rows, subplot_cols), (1, 0), colspan=1, xticklabels=[], yticklabels=[], fig=fig
)
ax1.plot(task_impedance_ts[:, 1], task_impedance_ts[:, 3], color=ts_color)
ax1.plot(joint_impedance_ts[:, 1], joint_impedance_ts[:, 3], color=js_color, linestyle="--")
ax1.set_xlabel("$e_x$", fontdict=font, va='center')
ax1.set_ylabel("$f_{int}$", fontdict=font, va='center')

joint = 1
ax2 = plt.subplot2grid(
    (subplot_rows, subplot_cols), (0, 0), colspan=1, xticklabels=[], yticklabels=[]
)
ax2.plot(
    rad2deg * task_impedance_js[:, joint],
    task_impedance_js[:, joint + 6],
    color=colormaps['tab20c'](1),
)
ax2.plot(
    rad2deg * joint_impedance_js[:, joint],
    joint_impedance_js[:, joint + 6],
    color=colormaps['tab20c'](5),
    linestyle="--",
)
ax2.set_xlabel("$q$", fontdict=font, va='center')
ax2.set_ylabel(r"$\tau$", fontdict=font, va='center')


ax3 = plt.subplot2grid(
    (subplot_rows, subplot_cols),
    (0, 1),
    rowspan=2,
    projection="3d",
    xticklabels=[],
    yticklabels=[],
    zticklabels=[],
)

ax3.set_proj_type("ortho")
ax3.plot(
    task_impedance_ts[:, 1],
    task_impedance_ts[:, 2],
    task_impedance_ts[:, 3],
    color=ts_color,
)
ax3.plot(
    joint_impedance_ts[:, 1],
    joint_impedance_ts[:, 2],
    joint_impedance_ts[:, 3],
    color=js_color,
    linestyle="--",
)
ax3.set_xlabel("$e_x$", fontdict=font)
ax3.set_ylabel("$\dot{e}_x$", fontdict=font)
ax3.set_zlabel("$f_{int}$", fontdict=font)
ax3.view_init(elev=20, azim=-20)

ax4 = plt.subplot2grid(
    (subplot_rows, subplot_cols), (2, 0), colspan=1, xticklabels=[], yticklabels=[]
)
ax4.plot(task_impedance_ts[:, 1], task_impedance_ts[:, 2], color=ts_color)
ax4.plot(joint_impedance_ts[:, 1], joint_impedance_ts[:, 2], color=js_color, linestyle="--")
ax4.set_ylabel("$\dot{e}_x$", fontdict=font, va='center')
ax4.set_xlabel("$e_x$", fontdict=font, va='center')

ax5 = plt.subplot2grid(
    (subplot_rows, subplot_cols), (2, 1), colspan=1, xticklabels=[], yticklabels=[]
)
ax5.plot(task_impedance_ts[:, 2], task_impedance_ts[:, 3], color=ts_color)
ax5.plot(joint_impedance_ts[:, 2], joint_impedance_ts[:, 3], color=js_color, linestyle="--")
ax5.set_ylabel("$f_{int}$", fontdict=font, va='center')
ax5.set_xlabel("$\dot{e}_x$", fontdict=font, va='center_baseline')

for ax in fig.get_axes():
    ax.tick_params(direction='in')

plt.subplots_adjust(
    left=0.06, bottom=0.09, right=0.94, top=0.98, wspace=0.16, hspace=0.35
)
plt.show(block=True)
