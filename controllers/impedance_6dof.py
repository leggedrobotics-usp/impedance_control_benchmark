#   Classical Impedance Control - 6 DoF: Translational + Rotational
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
from example_robot_data  import load

robot_name = 'ur5'
robot = load(robot_name)
viz   = MeshcatVisualizer()

robot.setVisualizer(viz)
robot.initViewer(open=True)
robot.loadViewerModel()

NQ, NV = robot.model.nq, robot.model.nv
end_effector = robot.model.getFrameId('ee_link')

q =   np.array([0.00, -1.00, 1.00, -pie, -pie/2, 0])
#q =   np.array([0, -1.10, 1.10, 0, pie/2, 0])
#q =   np.array([-pie/6, -1.10, 1.10, 0, pie/2-pie/6, 0])
dq =  np.zeros(robot.model.nv)
tau = np.zeros(robot.model.nq)

# Cartesian Impedance Control
Kp = np.eye(6) * 5000
Kd = np.sqrt(Kp) * 2
#x_desired = np.array([0.484, 0.500, 0.1])
# Initiate the x_des according to the q0
pin.forwardKinematics(robot.model, robot.data, q)
x_desired = robot.framePlacement(q, end_effector).translation 
v_desired = np.zeros(6)
a_desired = np.zeros(6)

# Rotational Impedance:
# Roll, Pitch, Yaw:
Ree_des = pin.rpy.rpyToMatrix(0, 0, 0)
Kp[3:,3:] = np.eye(3)*70 # reduce the rotational stiffness
Kd[3:,3:] = np.sqrt(Kp[3:,3:]) * 2

# Singularity Avoidance Potential
Vm, Vm_last, dVm = 0, 0, 0
k_sap  = 1
m_dyn0 = 1

# Nullspace joint compliance
Kns = np.eye(NQ) * 50
Dns = np.sqrt(Kns) * 2
q_desired = q

# Simulation parameters
sim_duration = 3.50 # [s]
sim_dt = 0.001      # [s]
sim_steps = int(sim_duration/sim_dt)
# Choose the impedance controller
controller = 3
singularity_avd = False
dynamic_ref = True

# Disturbance
disturbance_t = 0.5
ur5_payload = 5.00 # [Kg]
fe_amp  = ur5_payload*9.81
fe_angf = 2*pie*1.0
tau_ext   = np.zeros(NQ)
force_ext = np.array([0, 0, 0])
#static_x_expt = x_desired - np.linalg.inv(Kp).dot(force_ext)

delta_t = []
avg_freq = 0
check_jinv = []
nan_jinv = False
JTpinv = np.zeros((3, NQ)) # 

# Logging
plotting = True
log_fee = []
log_xee = []
log_vee = []

try:
    print(f'Simulation started, Controller {controller}. Press CTRL+C to stop.')
    tic = tm.time()
    for k in range(sim_steps):
        dt = sim_dt
        sim_time = k*dt

        ## Compute Terms ##
        # Compute: FK | crba | nle | Jacobians | CoM | K+U Energies...
        pin.computeAllTerms(robot.model, robot.data, q, dq)
        M = robot.mass(q)    # get the Inertia Matrix
        h = robot.nle(q, dq) # get Nonlinear Effects (C+g)
        g = robot.gravity(q) # get Gravitational vector
        Mi = pin.computeMinverse(robot.model, robot.data, q)

        # Get the end-effector Jacobian
        J = pin.computeFrameJacobian(robot.model, robot.data, 
                                      q, end_effector, pin.ReferenceFrame.LOCAL)

        # End-effector Jacobian derivative (dJ/dt)
        dJ = pin.frameJacobianTimeVariation(robot.model, robot.data, 
                                            q, dq, end_effector, pin.ReferenceFrame.LOCAL) 

        # Get the end-effector frame position and velocity w.r.t. the WF
        pin.updateFramePlacement(robot.model, robot.data, end_effector)   
        x = robot.data.oMf[end_effector].translation            # 3x1     
        v = J @ dq # using the definiton of J (numpy product @) # 6x1

        Ree = robot.data.oMf[end_effector].rotation
        rpy = pin.rpy.matrixToRpy(robot.data.oMf[end_effector].rotation) # useful to check the orientation

        if singularity_avd:
            # Singularity Avoidance Potential:
            m_dyn  = math.sqrt(np.linalg.det(J @ Mi @ Mi.T @ J.T)) # manipulability measure
            if m_dyn < m_dyn0:
                Vm = k_sap*(m_dyn - m_dyn0)**2
            else:
                Vm = 0
            # TODO: avoid divide by zero in this Finite Difference
            dVm = (Vm - Vm_last)/sim_dt * (1/dq)
            dVm.clip(-1000, 1000)
            Vm_last = Vm
        
        if dynamic_ref:
            ref_freq = 0.3
            ref_ampl = 0.10
            mov_ee_angle = 2*pie*ref_freq*sim_time
            x_desired[1] = ref_ampl * math.cos(mov_ee_angle) + 0.1091
            x_desired[2] = ref_ampl * math.sin(mov_ee_angle) + 0.5414
            v_desired[1] = -(2*pie*ref_freq) * ref_ampl * math.sin(mov_ee_angle)
            v_desired[2] =  (2*pie*ref_freq) * ref_ampl * math.cos(mov_ee_angle)
            a_desired[1] = -(2*pie*ref_freq)**2 * ref_ampl * math.cos(mov_ee_angle)
            a_desired[2] = -(2*pie*ref_freq)**2 * ref_ampl * math.sin(mov_ee_angle)
        
        ## Control ##
        # TODO: def functions for each controller       

        if controller == 0:     # No controller
            tau = np.zeros(NQ) + h
        
        if controller == 3:
            # [Ott,2008]: Classical cartesian impedance Eq. 3.18
            x_err = np.concatenate([x_desired -x, pin.rpy.matrixToRpy(Ree_des @ Ree.T)])
            v_err = v_desired -v
            force_impedance = Kp.dot(x_err) + Kd.dot(v_err)
            Ji = np.linalg.inv(J)
            tau = g + M @ J @ a_desired + (h-g -M @ Ji @ dJ) @ Ji @ v_desired + J.T @ force_impedance

        if controller == 4:
            # [Ott,2008]: Classical cartesian impedance with a_d = v_d = 0
            pos_err =  x_desired - x
            rpy_err = pin.rpy.matrixToRpy(Ree_des @ Ree.T)
            err = np.concatenate([pos_err, rpy_err])
            force_impedance = Kp.dot(err) + Kd.dot(-v)
            tau = g + J.T @ force_impedance

        if controller == 5:
            # Joint Space impedance (Tutorial):
            Kq = np.diag([1000, 1000, 500, 500, 100, 100])
            Dq = np.sqrt(Kq) * 2
            tau_impedance = Kq.dot(q_desired -q) + Dq.dot(-dq)
            tau = g + tau_impedance

        check_jinv.append(nan_jinv)
        
        #tau[0] = 0.5 * math.sin( 2 * pie * 0.4 * sim_time)
        #tau = tau + h -0.2*dq

        # Disturbance:
        if sim_time > disturbance_t:
            force_ext[0] = -fe_amp * math.sin(fe_angf * sim_time)
            tau_ext = J[:3,:].T @ force_ext
        
        ## Simulate Dynamics ##
        # Forward Dynamics (Articulated-Body Algorithm):
        ddq = pin.aba(robot.model, robot.data, q, dq, tau + tau_ext)
        # Integration:
        dq = dq + dt * ddq
        q  = pin.integrate(robot.model, q, dt*dq)

        viz.display(q)

        ## Logging ##
        log_fee.append(-force_ext[0])
        log_xee.append(x_desired[0] - x[0])
        log_vee.append(v_desired[0] - v[0])
        #log_fee.append(m_kin)
        #log_xee.append(sim_time)
    # Check how long it take to run the simulation
    ellapsed_time = tm.time() - tic
    print(f'\nSimulation ended. ({ellapsed_time:.2f} s)\n') # :.2f formatting float with 2 decimals
    print(any(check_jinv))

except KeyboardInterrupt:
    # Stop loop with CTRL+C
    print(any(check_jinv))
    exit

if plotting:
    plt.figure()
    plt.plot(log_xee, log_fee)
    plt.grid()
    plt.show(block=False) # figure wont close running in interactive mode
    # 3D plot:
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(log_xee, log_vee, log_fee)
    plt.show(block=False)


# We should solve the Nullspace and singularity separately
# About Nullspace projection:
# https://robotics.stackexchange.com/questions/23600/redundancy-and-null-space-projection
# About UR5 singularities:
# https://www.universal-robots.com/articles/ur/application-installation/what-is-a-singularity/
# SVD convergence is sensitve to initial condition (q0 must be different from 0) 

# Robot Model in Joint Space:
# M(q) * ddq + C(q,dq) * dq + g(q) = tau + tau_ext

# Robot Model in Task Space (Ott, 2008, item 3.1.2)
# Lambda(x) * ddx + u(x,dx) * dx + J(q)^{-T} * g(q) = J(q)^{-T} * tau + F_ext
# J(q)^{-T} * g(q) is also known as F_g(x) and J(q)^{-T} = F_tau
# Lambda(x) = J(q)^{-T} * M(q) * J(q)^{-1}
# u(x,dx)   = J(q)^{-T} * [C(q,dq) - M(q) * J(q)^{-1} * dJ(q)] * J(q)^{-1}
#
# Jacobian inverse can be avoided to be computed by the following identity:
# (A*B)^{-1} = B^-1 * A^-1 (A and B must have inverse...)
# Then:
# M(q) * J(q)^{-1} = [J(q) * M(q)^-1]^-1
# Lambda(x) = (J * M^-1 * J.T)^1

# # Compute the Damping Matrix (Ott, 2008, 3.3):
# # from scipy.linalg import eigh
# A = J @ np.linalg.inv(M)
# Lambda = np.linalg.inv(A @ J.T)
# Bo, B = eigh(Kp[:3, :3], Lambda)
# D = 2 * B.T @ np.diag(1.0*np.sqrt(Bo)) @ B # damping factor = 1.0
# Kd[:3,:3] = D