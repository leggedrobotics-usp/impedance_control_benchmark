import numpy as np
import math

#   Compute the transforms according to the system (Kd,Dd,Md) and input frequency 'wu'
def transforms(arg_dict):
    K = arg_dict.get("Kd")
    D = arg_dict.get("Dd")
    M = arg_dict.get("Md")
    wu = arg_dict.get("wu")

    binormal_vector = np.array([K - M * wu**2, D, -1])

    tht = math.atan(D)
    phi = math.asin(-binormal_vector[0] / np.linalg.norm(binormal_vector))

    Tx = np.array(
        [
            [1, 0, 0],
            [0, math.cos(tht), -math.sin(tht)],
            [0, math.sin(tht), math.cos(tht)],
        ]
    )
    Ty = np.array(
        [
            [math.cos(phi), 0, math.sin(phi)],
            [0, 1, 0],
            [-math.sin(phi), 0, math.cos(phi)],
        ]
    )
    
    sigma_1 = math.sqrt(D**2 + 1)
    sigma_2 = K - M*wu**2

    R11 = math.sqrt(1 + (sigma_2/sigma_1)**2)
    R12 = 0
    R21 = D * sigma_2 / sigma_1
    R22 = wu * sigma_1
    
    Tz = np.array([[R11, R12, 0], [R21, R22, 0], [0, 0, 1]])

    return Tx, Ty, Tz
