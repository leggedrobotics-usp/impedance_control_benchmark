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

import numpy as np
from math import sin, cos, atan, asin, sqrt

def transforms(arg_dict):
    """Compute the ellipse-based transforms representing a 2º order LTI system
    under sinusoidal input.

    Args:
        arg_dict (str, double): lumped parameters values dictionary:
        -- Kd: equivalent stiffness;
        -- Dd: equivalent damping;
        -- Md: equivalent inertia;
        -- wu: input angular frequency;

    Returns:
        np.array((3,3)), np.array((3,3)), np.array((3,2)): 
        sequential transformations Tx, Ty, and Tz, respectively.
    """
    K = arg_dict.get("Kd")
    D = arg_dict.get("Dd")
    M = arg_dict.get("Md")
    wu = arg_dict.get("wu")

    binormal_vector = np.array([K - M * wu**2, D, -1])

    rho = atan(D)
    phi = asin(-binormal_vector[0] / np.linalg.norm(binormal_vector))

    Tx = np.array(
        [
            [1, 0, 0],
            [0, cos(rho), -sin(rho)],
            [0, sin(rho),  cos(rho)],
        ]
    )
    Ty = np.array(
        [
            [cos(phi), 0, sin(phi)],
            [0, 1, 0],
            [-sin(phi), 0, cos(phi)],
        ]
    )
    
    sigma_1 = sqrt(D**2 + 1)
    sigma_2 = K - M*wu**2

    R11 = sqrt(1 + (sigma_2/sigma_1)**2)
    R12 = 0
    R21 = D * sigma_2 / sigma_1
    R22 = wu * sigma_1
    
    Tz = np.array([[R11, R12], [R21, R22], [0, 0]])

    return Tx, Ty, Tz
