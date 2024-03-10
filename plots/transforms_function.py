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
    
    # P11 = 0
    
    sigma_1 = math.sqrt(D**2 + 1)
    sigma_2 = K - M*wu**2
    sigma_3 = D**2 + abs(sigma_2)**2 + 1
    
    P12 = (
        math.sqrt(1 - sigma_2**2/sigma_3) +
        sigma_2**2/(sigma_1 * math.sqrt(sigma_3))
    )
    P21 = wu * sigma_1
    P22 = D * sigma_2 / sigma_1

    P = np.array([[0, P12, 0], [P21, P22, 0], [0, 0, 1]])

    return Tx, Ty, P
