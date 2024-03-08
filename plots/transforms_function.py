import numpy as np
import math


#   Compute the transforms according to the system (Kd,Dd,Md) and input frequency 'wu'
def transforms(arg_dict):
    Kd = arg_dict.get("Kd")
    Dd = arg_dict.get("Dd")
    Md = arg_dict.get("Md")
    wu = arg_dict.get("wu")

#TODO(qleonardolp): Update R definition using K,D,M

    wn = math.sqrt(Kd / Md)  # -> natural freq.
    zeta = Dd / (2 * Md * wn)  # -> damping ratio

    n_vec = np.array([Kd - Md * wu**2, Dd, -Kd])

    tht = math.atan(Dd / Kd)
    phi = math.asin(-n_vec[0] / np.linalg.norm(n_vec))

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

    R11 = -(
        2
        * wn**3
        * wu
        * zeta
        * (
            1
            - (wn**2 - wu**2) ** 2
            / (
                abs(wn**2 - wu**2) ** 2
                + 4 * wn**2 * abs(zeta) ** 2
                + wn**4 / Kd * Kd
            )
        )
        ** (1 / 2)
    ) / (
        Kd
        * (
            wn**4
            + 4 * wn**2 * wu**2 * zeta**2
            - 2 * wn**2 * wu**2
            + wu**4
        )
    ) - (
        2 * wn * wu * zeta * (wn**2 - wu**2) ** 2
    ) / (
        ((4 * Kd * Kd * zeta**2) / wn**2 + 1) ** (1 / 2)
        * (
            abs(wn**2 - wu**2) ** 2
            + 4 * wn**2 * abs(zeta) ** 2
            + wn**4 / Kd * Kd
        )
        ** (1 / 2)
        * (
            wn**4
            + 4 * wn**2 * wu**2 * zeta**2
            - 2 * wn**2 * wu**2
            + wu**4
        )
    )
    R12 = (
        ((wn**2 - wu**2))
        / (
            ((4 * Kd * Kd * zeta**2) / wn**2 + 1) ** (1 / 2)
            * (
                abs(wn**2 - wu**2) ** 2
                + 4 * wn**2 * abs(zeta) ** 2
                + wn**4 / Kd * Kd
            )
            ** (1 / 2)
        )
        + (
            wn**2
            * (wn**2 - wu**2)
            * (
                1
                - (wn**2 - wu**2) ** 2
                / (
                    abs(wn**2 - wu**2) ** 2
                    + 4 * wn**2 * abs(zeta) ** 2
                    + wn**4 / Kd * Kd
                )
            )
            ** (1 / 2)
        )
        / (
            Kd
            * (
                wn**4
                + 4 * wn**2 * wu**2 * zeta**2
                - 2 * wn**2 * wu**2
                + wu**4
            )
        )
        - (4 * wn**2 * wu**2 * zeta**2 * (wn**2 - wu**2))
        / (
            ((4 * Kd * Kd * zeta**2) / wn**2 + 1) ** (1 / 2)
            * (
                abs(wn**2 - wu**2) ** 2
                + 4 * wn**2 * abs(zeta) ** 2
                + wn**4 / Kd * Kd
            )
            ** (1 / 2)
            * (
                wn**4
                + 4 * wn**2 * wu**2 * zeta**2
                - 2 * wn**2 * wu**2
                + wu**4
            )
        )
    )
    R21 = (wn**3 * wu * (wn**2 - wu**2)) / (
        Kd
        * (4 * zeta**2 * Kd * Kd + wn**2) ** (1 / 2)
        * (
            wn**4
            + 4 * wn**2 * wu**2 * zeta**2
            - 2 * wn**2 * wu**2
            + wu**4
        )
    )
    R22 = (2 * Kd * zeta) / (4 * zeta**2 * Kd * Kd + wn**2) ** (1 / 2) + (
        2 * wn**4 * wu**2 * zeta
    ) / (
        Kd
        * (4 * zeta**2 * Kd * Kd + wn**2) ** (1 / 2)
        * (
            wn**4
            + 4 * wn**2 * wu**2 * zeta**2
            - 2 * wn**2 * wu**2
            + wu**4
        )
    )

    Tz = np.array([[R11, R12, 0], [R21, R22, 0], [0, 0, 1]])

    return Tx, Ty, Tz
