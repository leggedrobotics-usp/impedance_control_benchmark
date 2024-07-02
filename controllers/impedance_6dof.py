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

import pinocchio as pin
import numpy as np
import time as tm
import math
from math import pi
import matplotlib.pyplot as plt
from pinocchio.visualize import MeshcatVisualizer
import example_robot_data


controller = 1  # Choose the impedance controller
plotting = True  # dismiss plot if false
savefile = False  # dismiss log save if false
open_viewer = (
    False  # false if the viewer tab is already open (http://127.0.0.1:7000/static/)
)

robot = example_robot_data.load("ur5")
viz = MeshcatVisualizer()

robot.setVisualizer(viz)
robot.initViewer(open=open_viewer)
robot.loadViewerModel()
NQ, NV = robot.model.nq, robot.model.nv
end_effector = robot.model.getFrameId("ee_link")

q = np.array([0.00, -1.00, 1.00, -pi, -pi / 2, 0])
dq = np.zeros(robot.model.nv)
ddq = np.zeros(robot.model.nv)
tau = np.zeros(robot.model.nq)

sim_dt = 0.010  # [s] simulation time step

# Cartesian Impedance Control
# Rotational Impedance (Roll, Pitch, Yaw):
Ree_des = pin.rpy.rpyToMatrix(0, 0, 0)

if controller == 2:
    sim_dt = 0.001  # [s]
    Kd = np.diag([1000, 7000, 7000, 200, 200, 200])  # Stiffness
    wn = np.diag([250, 50, 50, 30, 30, 30])  # natural freq.
    Md = Kd @ np.linalg.inv(wn @ wn.T)  # M = K/w²
    Md_inv = np.linalg.inv(Md)
    # Damping design, D = 2 * sqrt(K*M) * zeta:
    zeta = 1.0
    Dd = 2 * np.sqrt(Kd @ Md) * zeta  # Damping
if controller == 3 or controller == 4:  # 1st order impedance
    Kd = np.diag([2500, 2500, 2500, 150, 150, 188])  # Stiffnes
    Dd = np.diag([125, 100, 100, 0.6, 0.6, 0.8])  # Damping
if controller == 1 or controller == 5:
    q_des = q
    dq_des = np.zeros(robot.model.nv)
    ddq_des = np.zeros(robot.model.nv)

# Simulation time parameters
sim_duration = 4.00  # [s]
sim_steps = int(sim_duration / sim_dt)

# Initiate the x_des according to the q0
pin.forwardKinematics(robot.model, robot.data, q)
x_desired = robot.framePlacement(q, end_effector).translation
v_desired = np.zeros(6)
a_desired = np.zeros(6)

# Singularity Avoidance Potential
Vm, Vm_last = 0, 0
dVm = np.zeros(NV)
k_sap = 1.00
m_dyn0 = 8.58e-2  # experimental threshold

singularity_avd = False
dynamic_ref = False

# Disturbance
disturbance_t = 0.01
ur5_payload = 5.00  # [Kg]
fe_amp = 9.80665 * ur5_payload
fe_angf = 3.00  # [Hz]
tau_ext = np.zeros(NQ)
force_ext = np.array([-fe_amp, 0, 0, 0, 0, 0])

delta_t = []
JTpinv = np.zeros((3, NQ))  #

# Logging
log_fee = []  # force_ext logging data
log_xee = []  # ee pos error logging data
log_vee = []  # ee vel error logging data
log_aee = []  # ee acc error logging data
log_time = []  # time logging data
# Log JS states
log_torque = []
log_joints = []

tau_id_full_j1 = []
tau_id_full_j3 = []

tau_id_part_j1 = []
tau_id_part_j3 = []

tau_nle_j1 = []
tau_nle_j3 = []

viz.display(q)
input(
    "Wait the visualizer loading on the browser window or reload it if already open.\n"
    + "Then, press any key here to start."
)
try:
    print(f"Simulation started, Controller {controller}. Press CTRL+C to stop.")
    tic = tm.time()
    for k in range(sim_steps):
        dt = sim_dt
        sim_time = k * dt

        ## Compute Terms ##
        # Compute: FK | crba | nle | Jacobians | CoM | K+U Energies
        pin.computeAllTerms(robot.model, robot.data, q, dq)
        M = robot.mass(q)  # get the Inertia Matrix
        h = robot.nle(q, dq)  # get Nonlinear Effects (C+g)
        g = robot.gravity(q)  # get Gravitational vector

        # Get the end-effector Jacobian
        J = pin.computeFrameJacobian(
            robot.model, robot.data, q, end_effector, pin.ReferenceFrame.LOCAL
        )

        # End-effector Jacobian derivative (dJ/dt)
        dJ = pin.frameJacobianTimeVariation(
            robot.model, robot.data, q, dq, end_effector, pin.ReferenceFrame.LOCAL
        )

        # Get ee kinematic data:
        pin.updateFramePlacement(robot.model, robot.data, end_effector)
        x = robot.data.oMf[end_effector].translation  # 3x1
        v = J @ dq
        ddx = dJ @ dq + J @ ddq

        Ree = robot.data.oMf[end_effector].rotation
        rpy = pin.rpy.matrixToRpy(
            robot.data.oMf[end_effector].rotation
        )  # useful for orientation check

        if singularity_avd:
            dVm = np.zeros(NV)
            # Singularity Avoidance Potential:
            m_dyn = math.sqrt(np.linalg.det(J @ J.T))  # manipulability measure
            if m_dyn < m_dyn0:
                Vm = k_sap * (m_dyn - m_dyn0) ** 2
                for k in range(NV):
                    if dq[k] > 1e-6:
                        dVm[k] = (Vm - Vm_last) / (dq[k] * sim_dt)
                    else:
                        dVm[k] = (1.0 - 1e-6) * dVm[k]
            else:
                Vm = 0
            Vm_last = Vm

        if dynamic_ref:
            ref_freq = 0.3
            ref_ampl = 0.10
            mov_ee_angle = 2 * pi * ref_freq * sim_time
            x_desired[1] = ref_ampl * math.cos(mov_ee_angle) + 0.1091
            x_desired[2] = ref_ampl * math.sin(mov_ee_angle) + 0.5414
            v_desired[1] = -(2 * pi * ref_freq) * ref_ampl * math.sin(mov_ee_angle)
            v_desired[2] = (2 * pi * ref_freq) * ref_ampl * math.cos(mov_ee_angle)
            a_desired[1] = (
                -((2 * pi * ref_freq) ** 2) * ref_ampl * math.cos(mov_ee_angle)
            )
            a_desired[2] = (
                -((2 * pi * ref_freq) ** 2) * ref_ampl * math.sin(mov_ee_angle)
            )

        # Disturbance:
        if sim_time > disturbance_t:
            force_ext[0] = -fe_amp * math.sin(fe_angf * sim_time)
            tau_ext = J.T @ force_ext

        ## Control ##
        x_err = np.concatenate([x_desired - x, pin.rpy.matrixToRpy(Ree_des @ Ree.T)])
        v_err = v_desired - v

        if controller == 0:  # No controller
            tau = np.zeros(NQ) + h

        if controller == 1:
            # [T. Boaventura, 2012]: Joint Space PD
            Kp = 100 * np.eye(NQ)
            Kd = 20 * np.eye(NQ)
            joint_impedance = Kp.dot(q_des - q) + Kd.dot(dq_des - dq)
            tau_id_part_j1.append(joint_impedance[0])
            tau_id_part_j3.append(joint_impedance[2])
            joint_impedance = M @ joint_impedance
            tau_id_full_j1.append(joint_impedance[0])
            tau_id_full_j3.append(joint_impedance[2])
            tau = M @ ddq_des + h + joint_impedance

        if controller == 2:
            # [Ott,2008]: Classical Impedance Controler Eq. 3.14
            Ji = np.linalg.inv(J)
            impedance_force = Kd.dot(x_err) + Dd.dot(v_err)
            IR = M @ Md_inv  # Inertial Ratio
            interaction_port = (J.T @ IR) @ (
                impedance_force + force_ext
            ) - J.T @ force_ext
            inverse_dyn = M @ J @ a_desired + (h - g - M @ Ji @ dJ) @ Ji @ v_desired
            tau = inverse_dyn + interaction_port

        if controller == 3:
            # [Ott,2008]: Classical Impedance Controler (No Inertia) Eq. 3.18
            Ji = np.linalg.inv(J)
            impedance_force = Kd.dot(x_err) + Dd.dot(v_err)
            interaction_port = J.T @ impedance_force
            inverse_dyn = M @ J @ a_desired + (h - g - M @ Ji @ dJ) @ Ji @ v_desired
            tau = inverse_dyn + interaction_port

        if controller == 4:
            # [Ott,2008]: Classical Impedance Controler (No Inertia), and a_d = v_d = 0
            impedance_force = Kd.dot(x_err) + Dd.dot(v_err)
            tau = J.T @ impedance_force

        if controller == 5:
            # Joint Space impedance (Tutorial):
            Kq = np.diag([1000, 1000, 1000, 500, 500, 200])
            wq = np.eye(6) * 250
            Mq = Kq @ np.linalg.inv(wq @ wq.T)
            Dq = 2 * np.sqrt(Kq @ Mq)  # zeta = 1
            tau_impedance = (
                Kq.dot(q_des - q) + Dq.dot(dq_des - dq) + Mq.dot(ddq_des - ddq)
            )
            tau = tau_impedance

        ## Simulate Dynamics ##
        # Forward Dynamics (Articulated-Body Algorithm):
        ddq = pin.aba(robot.model, robot.data, q, dq, tau + g + tau_ext + dVm)
        # Integration:
        dq = dq + dt * ddq
        q = pin.integrate(robot.model, q, dt * dq)

        viz.display(q)

        ## Logging ##
        log_time.append(sim_time)
        log_fee.append(-force_ext[0])
        log_xee.append(x_desired[0] - x[0])
        log_vee.append(v_desired[0] - v[0])
        log_aee.append(a_desired[0] - ddx[0])

        log_joints.append(q)
        log_torque.append(tau)

        tau_nle_j1.append(g[0] + h[0])
        tau_nle_j3.append(g[2] + h[2])

    ellapsed_time = tm.time() - tic
    print(f"\nSimulation ended. ({ellapsed_time:.2f} s)\n")

except KeyboardInterrupt:
    exit

if plotting:
    plt.figure()
    plt.plot(log_xee[2:], log_fee[2:])
    plt.grid()
    # figure wont close running in interactive mode
    plt.show(block=False)

    ax = plt.figure().add_subplot(projection="3d")
    ax.plot(log_xee[2:], log_vee[2:], log_fee[2:])
    plt.show(block=(not open_viewer))

if savefile:
    np.save("data/ts_data", np.array([log_time, log_xee, log_vee, log_fee, log_aee]).T)
    time = np.array([log_time]).T
    joints = np.array(log_joints)
    torque = np.array(log_torque)
    js_log = np.hstack((time, joints, torque))
    np.save("data/js_data", js_log)
    if controller == 1:
        np.save(
            "data/js_terms",
            np.array(
                [
                    log_time,
                    tau_nle_j1,
                    tau_id_part_j1,
                    tau_id_full_j1,
                    tau_nle_j3,
                    tau_id_part_j3,
                    tau_id_full_j3,
                ]
            ).T,
        )
