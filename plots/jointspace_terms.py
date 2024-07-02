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

js_terms = numpy.load("data/js_terms.npy")
time = js_terms[:,0]

id_full = colormaps['tab20c'](0)
id_partial = colormaps['tab20c'](4)
grav_comp = colormaps['tab20c'](17)
font = {
    "family": "serif",
    "math_fontfamily": "cm",
    "color": "black",
    "weight": "bold",
    "size": 13,
}

subplot_cols = 2
subplot_rows = 1

fig = plt.figure(figsize=[9.51, 3.74])

ax1 = plt.subplot2grid((subplot_rows, subplot_cols), (0, 0), colspan=1, fig=fig)
ax1.plot(time, js_terms[:,1], color=grav_comp, linestyle="-.")
ax1.plot(time, js_terms[:,2], color=id_partial, linestyle="--")
ax1.plot(time, js_terms[:,3], color=id_full, linestyle="-")

ax2 = plt.subplot2grid((subplot_rows, subplot_cols), (0, 1), colspan=1, fig=fig)
ax2.plot(time, js_terms[:,4], color=grav_comp, linestyle="-.")
ax2.plot(time, js_terms[:,5], color=id_partial, linestyle="--")
ax2.plot(time, js_terms[:,6], color=id_full, linestyle="-")

for ax in fig.get_axes():
    ax.tick_params(direction='in')
    ax.set_xlabel("$time \hspace{0.5} [s]$", fontdict=font, va='center')
    ax.set_ylabel(r"$\tau \hspace{0.5} [N.m]$", fontdict=font, va='center')

plt.subplots_adjust(left=0.06, bottom=0.09, right=0.94, top=0.98, wspace=0.16, hspace=0.70)
plt.show(block=True)
