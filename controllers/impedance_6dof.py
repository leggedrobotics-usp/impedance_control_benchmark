#   Classical Impedance Controller - 6 DoF: Translational + Rotational
#   Leonardo F. dos Santos, 2023 | qleonardolp
#   REMEMBER: run on interactive mode (MeshcatVisualizer):
#   python3 -i impedance_6dof.py

import pinocchio as pin
import numpy as np
import time as tm
import math
from math import pi as pie
import matplotlib.pyplot as plt


from pinocchio.visualize import MeshcatVisualizer
from example_robot_data import load

robot_name = "ur5"
robot = load(robot_name)
viz = MeshcatVisualizer()

robot.setVisualizer(viz)
robot.initViewer(open=True)
robot.loadViewerModel()

NQ, NV = robot.model.nq, robot.model.nv
end_effector = robot.model.getFrameId("ee_link")

q = np.array([0.00, -1.00, 1.00, -pie, -pie / 2, 0])
dq = np.zeros(robot.model.nv)
ddq = np.zeros(robot.model.nv)
tau = np.zeros(robot.model.nq)

# Cartesian Impedance Control
Kd = np.diag([1000, 7000, 7000, 200, 200, 200])  # Stiffness
wn = np.diag([250, 50, 50, 30, 30, 30])  # natural freq.
Md = Kd @ np.linalg.inv(wn @ wn.T)  # M = K/w²
# Md = np.diag([9.2, 9.2, 9.2, .18, .18, .18])    # Inertia
Md_inv = np.linalg.inv(Md)
# Rotational Impedance (Roll, Pitch, Yaw):
Ree_des = pin.rpy.rpyToMatrix(0, 0, 0)
# Damping design, D = 2 * sqrt(K*M) * zeta:
zeta = 1.0  # 0.5 | 0.7 | 1.0
Dd = 2 * np.sqrt(Kd @ Md) * zeta  # Damping

# Initiate the x_des according to the q0
pin.forwardKinematics(robot.model, robot.data, q)
x_desired = robot.framePlacement(q, end_effector).translation
v_desired = np.zeros(6)
a_desired = np.zeros(6)

# Singularity Avoidance Potential
Vm, Vm_last = 0, 0
dVm = np.zeros(NV)
k_sap = 1.00
m_dyn0 = 8.58e-2  # experimentally computed threshold

# Nullspace joint compliance
Kns = np.eye(NQ) * 50
Dns = np.sqrt(Kns) * 2
q_des = q
dq_des = np.zeros(robot.model.nv)
ddq_des = np.zeros(robot.model.nv)

# Simulation parameters
sim_duration = 4.00  # [s]
sim_dt = 0.001  # [s]
sim_steps = int(sim_duration / sim_dt)
# Choose the impedance controller
controller = 2
singularity_avd = False
dynamic_ref = False

# Disturbance
disturbance_t = 0.01
ur5_payload = 5.00  # [Kg]
fe_amp = 9.80665 * ur5_payload
fe_angf = 3.00  # [Hz]
tau_ext = np.zeros(NQ)
force_ext = np.array(
    [-fe_amp, 0, 0, 0, 0, 0]
)  # somehow the plots are more beautiful with -fe_amp here (??)
# static_x_expt = x_desired - np.linalg.inv(Kd).dot(force_ext)

delta_t = []
JTpinv = np.zeros((3, NQ))  #

# Logging
plotting = True
savefile = False
log_fee = []  # force_ext logging data
log_xee = []  # ee pos error logging data
log_vee = []  # ee vel error logging data
log_aee = []  # ee acc error logging data
log_time = []  # time logging data
# Log JS states
log_torque = []
log_joints = []

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
        v = J @ dq  # using J definition (numpy product @)       # 6x1
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
            mov_ee_angle = 2 * pie * ref_freq * sim_time
            x_desired[1] = ref_ampl * math.cos(mov_ee_angle) + 0.1091
            x_desired[2] = ref_ampl * math.sin(mov_ee_angle) + 0.5414
            v_desired[1] = -(2 * pie * ref_freq) * ref_ampl * math.sin(mov_ee_angle)
            v_desired[2] = (2 * pie * ref_freq) * ref_ampl * math.cos(mov_ee_angle)
            a_desired[1] = (
                -((2 * pie * ref_freq) ** 2) * ref_ampl * math.cos(mov_ee_angle)
            )
            a_desired[2] = (
                -((2 * pie * ref_freq) ** 2) * ref_ampl * math.sin(mov_ee_angle)
            )

        # Disturbance:
        if sim_time > disturbance_t:
            force_ext[0] = -fe_amp * math.sin(fe_angf * sim_time)
            tau_ext = J.T @ force_ext

        ## Control ##

        if controller == 0:  # No controller
            tau = np.zeros(NQ) + h

        if controller == 2:
            # [Ott,2008]: Classical Impedance Controler Eq. 3.14
            x_err = np.concatenate(
                [x_desired - x, pin.rpy.matrixToRpy(Ree_des @ Ree.T)]
            )
            v_err = v_desired - v
            impedance_force = Kd.dot(x_err) + Dd.dot(v_err)
            Ji = np.linalg.inv(J)
            IR = M @ Md_inv  # Inertial Ratio
            tau = (
                M @ J @ a_desired
                + (h - g - M @ Ji @ dJ) @ Ji @ v_desired
                + (J.T @ IR) @ (impedance_force + force_ext)
                - J.T @ force_ext
            )
            # g(q) is on Forward Dynamics line

        if controller == 3:
            # [Ott,2008]: Classical Impedance Controler (No Inertia) Eq. 3.18
            x_err = np.concatenate(
                [x_desired - x, pin.rpy.matrixToRpy(Ree_des @ Ree.T)]
            )
            v_err = v_desired - v
            impedance_force = Kd.dot(x_err) + Dd.dot(v_err)
            Ji = np.linalg.inv(J)
            tau = (
                M @ J @ a_desired
                + (h - g - M @ Ji @ dJ) @ Ji @ v_desired
                + J.T @ impedance_force
            )
            # g(q) is on Forward Dynamics line

        if controller == 4:
            # [Ott,2008]: Classical Impedance Controler with a_d = v_d = 0
            pos_err = x_desired - x
            rpy_err = pin.rpy.matrixToRpy(Ree_des @ Ree.T)
            err = np.concatenate([pos_err, rpy_err])
            impedance_force = Kd.dot(err) + Dd.dot(-v)
            tau = J.T @ impedance_force
            # g(q) is on Forward Dynamics line

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
            # g(q) is on Forward Dynamics line

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

    ellapsed_time = tm.time() - tic
    print(f"\nSimulation ended. ({ellapsed_time:.2f} s)\n")

except KeyboardInterrupt:
    exit

if plotting:
    plt.figure()
    plt.plot(log_xee, log_fee)
    plt.grid()
    # figure wont close running in interactive mode
    plt.show(block=False)

    ax = plt.figure().add_subplot(projection="3d")
    ax.plot(log_xee, log_vee, log_fee)
    plt.show(block=False)

if savefile:
    # TODO(qleonardolp): data and hour on the file name
    np.save("data/ts_data", np.array([log_time, log_xee, log_vee, log_fee, log_aee]).T)

    time = np.array([log_time]).T
    joints = np.array(log_joints)
    torque = np.array(log_torque)
    js_log = np.hstack((time, joints, torque))
    np.save("data/js_data", js_log)
    # use np.load to import data on the plotting script
    print("Logs saved!")
