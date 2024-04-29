import numpy as np
from math import sin, cos, atan, asin, sqrt

def transforms(arg_dict):
    """Compute the ellipse-based transforms for a 2º Order LTI impedance controller
    acting on a single mass Mr. The ellipse exist only for sinusoidal inputs.

    Args:
        arg_dict (str, double): lumped parameters values dictionary:
        -- Kd: impedance stiffness;
        -- Dd: impedance damping;
        -- Md: impedance inertia;
        -- wi: input angular frequency;
        -- Mr: actual mass;

    Returns:
        np.array((3,3)), np.array((3,3)), np.array((3,2)): 
        sequential transformations Tx, Ty, and Tz, respectively.
    """
    K = arg_dict.get("Kd")
    D = arg_dict.get("Dd")
    M = arg_dict.get("Md")
    wi = arg_dict.get("wi")
    Mr = arg_dict.get("Mr")

    binormal_vector = np.array([K - M * wi**2, D, -1])

    rho = atan(D)
    phi = asin(-binormal_vector[0] / np.linalg.norm(binormal_vector))

    Tx = np.array(
        [
            [1, 0, 0],
            [0, cos(rho), -sin(rho)],
            [0, sin(rho), cos(rho)],
        ]
    )
    Ty = np.array(
        [
            [cos(phi), 0, sin(phi)],
            [0, 1, 0],
            [-sin(phi), 0, cos(phi)],
        ]
    )

    sigma_03 = K**2 + D**2 + 1 - 2 * K * M * wi**2 + (M*wi**2)**2
    sigma_02 = K * (K - 2 * (M + Mr) * wi**2) + (
        (D * wi)**2 + (M*wi**2)**2 + (Mr*wi**2)**2 + 2 * M * Mr * wi**4
    )
    sigma_01 = D**2 + 1

    R11 = D*wi*sigma_03 / (sqrt(sigma_01) * sqrt((K - M * wi**2)**2 + D**2 + 1) * sigma_02)
    R12 = ((M + Mr) * wi**2 - K) * sqrt(sigma_03) / (sqrt(sigma_01) * sigma_02)
    R21 = wi * ((M + Mr*sigma_01) * wi**2 - K) / (sqrt(sigma_01) * sigma_02)
    R22 = D/sqrt(sigma_01) * (wi**2 * (Mr*((Mr + M)*wi**2 - K) - 1) / sigma_02 - 1)
    Tz = np.array([[R11, R12], [R21, R22], [0, 0]]) * Mr * wi**2
    return Tx, Ty, Tz
