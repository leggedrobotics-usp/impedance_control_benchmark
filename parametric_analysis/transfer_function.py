from transforms_function import *
import matplotlib.pyplot as plt
import numpy as np
import control
from math import pi, sin, cos

# Lumped parameters:
wi = 6.2831853  # input angular frequency
Ai = 50.00  # input amplitude
Mr = 1.000  # actual mass
Kd = 100.0  # impedance stiffness
Dd = 20.00  # impedance damping
Md = 0.100  # impedance inertia

#print(math.sqrt(Kd/Md))
parameters_dict = {"Kd": Kd, "Dd": Dd, "Md": Md, "wi": wi, "Mr": Mr}

input_period = 2 * pi / wi
time = np.linspace(0, 3*input_period, 600)
x_ref_time = Ai * np.sin(wi * time)

ellipse = []
Tx, Ty, Tz = transforms(parameters_dict)
for k in np.linspace(0, 2*pi, 256):
  ellipse.append(Ai * Tx @ Ty @ Tz @ np.array([sin(k), cos(k)]).T)
ellipse = np.array(ellipse)

s = control.tf('s')    # s-domain variable
Z = Md*s**2 + Dd*s + Kd # s-domain impedance
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

# control.step_response(x_error_laplace).plot()
error_response = control.forced_response(ref_to_error_tf, time, x_ref_time, transpose=True)
error_derivative = np.gradient(error_response.x[0], error_response.t)

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.set_proj_type("ortho")
ax1.plot(error_response.x[0], error_derivative, error_response.u[0])
ax1.plot(ellipse[:, 0], ellipse[:, 1], ellipse[:, 2],
  color="k", linestyle="--",
)
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(error_response.t, error_response.x[0], label="position error")
ax2.plot(error_response.t, error_derivative, label="velocity error")
ax2.plot(error_response.t, error_response.u[0], label="$f_{int}$")
plt.show(block=True)
