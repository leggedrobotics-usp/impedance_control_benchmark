import numpy as np
import math

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

    rho = math.atan(D)
    phi = math.asin(-binormal_vector[0] / np.linalg.norm(binormal_vector))

    Tx = np.array(
        [
            [1, 0, 0],
            [0, math.cos(rho), -math.sin(rho)],
            [0, math.sin(rho), math.cos(rho)],
        ]
    )
    Ty = np.array(
        [
            [math.cos(phi), 0, math.sin(phi)],
            [0, 1, 0],
            [-math.sin(phi), 0, math.cos(phi)],
        ]
    )
    
    sigma_10 = (D**2 + 1)
    sigma_09 = 2 * M * Mr * wi**4
    sigma_08 = Mr**2 * wi**4
    sigma_07 = (D * wi)**2
    sigma_06 = M**2 * wi**4
    sigma_05 = K - M * wi**2
    sigma_04 = 2 * K * M * wi**2
    sigma_03 = math.sqrt(sigma_10) * (
        K**2 - 2 * K * (M + Mr) * wi**2 +
        sigma_06 + sigma_07 + sigma_08 + sigma_09
    )
    sigma_02 = K**2 - sigma_04 + sigma_06 + (
        math.sqrt(sigma_10) * math.sqrt(
            D**2 - K**2 + sigma_04 - sigma_06 + abs(sigma_05)**2 + 1)
    )
    
    sigma_01 = math.sqrt(sigma_10) * math.sqrt(sigma_05**2 + sigma_10) * (
        K**2 - 2 * K * Mr * wi**2 - sigma_04 + sigma_06 + sigma_07 + sigma_08 + sigma_09
    )
    
    R11 = Mr * wi**2 * ((M + Mr) * wi**2 - K) * sigma_02 / sigma_01
    R12 = D * Mr * wi**3 * sigma_02 / sigma_01
    R21 = D * Mr * wi**2 * (
        wi**2 * (Mr*((Mr + M)*wi**2 - K) - 1) / sigma_03
        - 1 / math.sqrt(sigma_10)
    )
    R22 = Mr * wi**3 * ((M + Mr*sigma_10) * wi**2 - K) / sigma_03
    
    Tz = np.array([[R11, R12], [R21, R22], [0, 0]])

    return Tx, Ty, Tz
