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


controller = 0  # Choose the impedance controller
open_viewer = (
    False  # false if the viewer tab is already open (http://127.0.0.1:7000/static/)
)

robot = example_robot_data.load("hyq")
viz = MeshcatVisualizer()

robot.setVisualizer(viz)
robot.initViewer(open=open_viewer)
robot.loadViewerModel()
NQ, NV = robot.model.nq, robot.model.nv
# foot frames (left/right, front/hind): lf_foot, rf_foot, lh_foot, rh_foot
end_effector = robot.model.getFrameId("lf_foot")

# q(0,1,2) -> trunk translations
# q(3,4,5,6) -> trunk rotations (quat ?)
# q(7,8,9) -> left front leg
# q(10,11,12) -> left hind leg
# q(13,14,15) -> right front leg
# q(16,17,18) -> rigth hind leg

q = np.zeros(NQ)
q[2] =  0.577 # trunk height (z)
q[7] = -0.15
q[8] =  0.8
q[9] = -1.6

q[10] = -0.15
q[11] = -0.8
q[12] =  1.6

q[13] = -0.15
q[14] =  0.8
q[15] = -1.6

q[16] = -0.15
q[17] = -0.8
q[18] =  1.6

q0 = q

dq = np.zeros(NV)
ddq = np.zeros(NV)
tau = np.zeros(NV)

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
    Kd = np.diag([900, 2500, 2500, 150, 150, 188])  # Stiffnes
    Dd = np.diag([60, 100, 100, 0.6, 0.6, 0.8])  # Damping
if controller == 1 or controller == 5:
    q_des = q
    dq_des = np.zeros(robot.model.nv)
    ddq_des = np.zeros(robot.model.nv)

# Initiate the x_des according to the q0
pin.forwardKinematics(robot.model, robot.data, q)
x_desired = robot.framePlacement(q, end_effector).translation # 3x1
v_desired = np.zeros(3)

# Disturbance
disturbance_t = 0.01
fe_amp = 9.80665 * 1
fe_angf = 0.25  # [Hz]
tau_ext = np.zeros(NV)
force_ext = np.array([0, 0, 0, 0, 0, 0])

# Simulation time parameters
sim_duration = 4.0 + disturbance_t  # [s]
sim_steps = int(sim_duration / sim_dt)

# Joint space controller gain matrices
Kp_q = np.eye(9) * 188
Kd_q = np.eye(9) * 2.7

# Cartesian impedance controller gain matrices
Kd = np.diag([250, 250, 250])  # Stiffnes
Dd = np.diag([10.0, 10.0, 10.0])  # Damping

delta_t = []

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
            robot.model, robot.data, q, end_effector, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        # End-effector Jacobian derivative (dJ/dt)
        dJ = pin.frameJacobianTimeVariation(
            robot.model, robot.data, q, dq, end_effector, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        # Get ee kinematic data:
        pin.updateFramePlacement(robot.model, robot.data, end_effector)
        x = robot.data.oMf[end_effector].translation  # 3x1
        v = J @ dq
        
        # Disturbance:
        if sim_time > disturbance_t:
            #force_ext[2] = fe_amp * math.sin(fe_angf * sim_time)
            force_ext[1] = 10
            tau_ext = J.T @ force_ext

        # joint space controller on the other legs
        tau_js = Kp_q.dot(q0[10:] - q[10:]) + Kd_q.dot(np.zeros(9,) - dq[9:])
        tau = np.concatenate([np.zeros(9,), tau_js])
        
        # [Ott,2008]: Classical Impedance Controler (No Inertia), and a_d = v_d = 0
        x_err = x_desired - x
        v_err = v_desired - v[:3]
        impedance_force = Kd.dot(x_err) + Dd.dot(v_err)
        tau[6:9] = J[:3,6:9].T @ impedance_force
        
        #Kd = np.diag([40, 40, 40])
        #Dd = np.diag([0.1, 0.1, 0.1])
        #tau[6:9] = Kd.dot(q0[7:10] - q[7:10]) + Dd.dot(-dq[7:10])
    
        ## Restrict the dynamic simulation to one leg only (left front)
        tau_ext[0:6] = np.zeros(6,)
        tau_ext[9:]  = np.zeros(9,)

        ## Simulate Dynamics ##
        # Forward Dynamics (Articulated-Body Algorithm):
        ddq = pin.aba(robot.model, robot.data, q, dq, tau + g + tau_ext)
        # Integration:
        dq = dq + dt * ddq
        q = pin.integrate(robot.model, q, dt * dq)

        viz.display(q)

    ellapsed_time = tm.time() - tic
    print(f"\nSimulation ended. ({ellapsed_time:.2f} s)\n")

except KeyboardInterrupt:
    exit
