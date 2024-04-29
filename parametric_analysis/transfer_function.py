from transforms_function import *
import matplotlib.pyplot as plt
import numpy as np
import control
from math import pi, sin, cos

# Lumped parameters:
wi = 2*pi*1.2    # input *angular frequency
Ai = 0.679       # input amplitude
Mr = 2.718       # actual mass
Kd = 400.0       # impedance stiffness
Dd = 2*sqrt(Kd)  # impedance damping
Md = 0 # impedance inertia

#print(math.sqrt(Kd/Md))
parameters_dict = {"Kd": Kd, "Dd": Dd, "Md": Md, "wi": wi, "Mr": Mr}

input_period = 2 * pi / wi
time = np.linspace(0, 2.0*input_period, 600)
x_ref_time = Ai * np.sin(wi * time)

ellipse = []
Tx, Ty, Tz = transforms(parameters_dict)
for k in np.linspace(0, 2*pi, 180):
  ellipse.append(Ai * Tx @ Ty @ Tz @ np.array([cos(k), sin(k)]).T)
ellipse = np.array(ellipse)

s = control.tf('s')      # s-domain variable
Z = Md*s**2 + Dd*s + Kd  # s-domain impedance
x_ref_laplace = Ai * wi/(s**2 + wi**2) # equilibrium point sinusoidal transfer function
ref_to_error_tf = (Mr * s**2)/(Z + Mr * s**2)

"""Block diagram:
                       _____     _________
    x_ref --> + O --> | Z(s)|-->|  __1__  |__
               -^     |_____|   |__Mr*s^2_|  |   
              x |                            |
                |____________________________|
"""
x_error_laplace = ref_to_error_tf * x_ref_laplace

# control.step_response(x_error_laplace)
error_response = control.forced_response(ref_to_error_tf, time, x_ref_time, transpose=True)
error_derivative = np.gradient(error_response.x[0], error_response.t)
error_2nd_derivative = np.gradient(error_derivative, error_response.t)
force_int_time = Md * error_2nd_derivative + Dd * error_derivative + Kd * error_response.x[0]

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1, projection="3d")

ax1.set_proj_type("ortho")
ax1.plot(error_response.x[0], error_derivative, force_int_time)
ax1.plot(ellipse[:, 0], ellipse[:, 1], ellipse[:, 2], color="r", linestyle="--")
ax1.set_xlabel("$e$")
ax1.set_ylabel("$de$")
ax1.set_zlabel("$f_{int}$")

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(error_response.t, error_response.x[0], label="$e$")
ax2.plot(error_response.t, error_derivative, label="$\dot{e}$")
ax2.plot(error_response.t, force_int_time, label="$f^d_{int}$")
ax2.plot(error_response.t, error_response.u[0], label="$x_{ref}$", color="k", linestyle="--")
plt.legend()
plt.show(block=True)
