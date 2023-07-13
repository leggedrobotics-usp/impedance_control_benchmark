#   This code intends to simple execution of a impedance control on a robotic manipulator
#   and assure the pinocchio functionallity prior to further developments.
#   It means to be as simple as possible: no logging, no plots, minimal variables... 
#   Leonardo F. dos Santos, 2023 | qleonardolp
#   REMEMBER: run on interactive mode: 
#   python3 -i impedance_control_simple.py
#   Simulation perfomance is limited, however the computation of J and dJ
#    may require high sample rate. Consider wokring with 1000 Hz.

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
v_desired = np.zeros(3)
a_desired = np.zeros(3)

# Rotational Impedance:
# Roll, Pitch, Yaw:
Ree_des = pin.rpy.rpyToMatrix(0, 0, 0)
Kp[3:,3:] = np.eye(3)*70 # reduce the rotational stiffness
Kd[3:,3:] = np.sqrt(Kp[3:,3:]) * 2

# Nullspace joint compliance
Kns = np.eye(NQ) * 50
Dns = np.sqrt(Kns) * 2
q_desired = q

# Simulation parameters
sim_duration = 3.50 # [s]
sim_dt = 0.001      # [s] # maybe Jinv stability depends of sim_dt 
sim_steps = int(sim_duration/sim_dt)
# Choose the impedance controller
#'tboaventura' # or 'mmayr'
controller = 4

# Disturbance
disturbance_t = 0.5
ur5_payload = 5.00 # [Kg]
fe_amp  = ur5_payload*9.81
fe_angf = 2*pie*1.0
tau_ext   = np.zeros(NQ)
force_ext = np.array([-fe_amp, 0, 0])
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
#
# M(q) * J(q)^{-1} = [J(q) * M(q)^-1]^-1
# Lambda(x) = (J * M^-1 * J.T)^1

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

        # Get the end-effector Jacobian w.r.t. the WF
        J6 = pin.computeFrameJacobian(robot.model, robot.data, 
                                      q, end_effector, pin.ReferenceFrame.LOCAL)  
        # Take the first 3 rows of J6 cause we have a point contact            
        J = J6[:3, :]

        # End-effector Jacobian derivative (dJ/dt)
        dJ = pin.frameJacobianTimeVariation(robot.model, robot.data, 
                                            q, dq, end_effector, 
                                            pin.ReferenceFrame.LOCAL)[:3, :] 

        # Get the end-effector frame position and velocity w.r.t. the WF
        pin.updateFramePlacement(robot.model, robot.data, end_effector)   
        x = robot.data.oMf[end_effector].translation              
        v = J @ dq # using the definiton of J (numpy product @)

        Ree = robot.data.oMf[end_effector].rotation
        rpy = pin.rpy.matrixToRpy(robot.data.oMf[end_effector].rotation)

        # TODO: Singularity avoidance: ...
        # Chain rule: dJ/dq * dq/dt = dJ/dt
        
        ## Control ##
        # TODO: def functions for each controller       

        if controller == 0:     # No controller
            tau = np.zeros(NQ) + h
            
        if controller == 1:     # Cartesian Impedance with JS dynamics (T. Boaventura)
            # Must be fixed (see the Nullspace proj)
            Jpinv = np.linalg.inv(J.T @ J) @ J.T #numpy.linalg.LinAlgError: SVD did not converge
            if not np.isnan(Jpinv).any():
                # Cartesian Impedance Control
                M_Jinv = np.matmul(M, Jpinv)
                force_impedance = Kp.dot(x_desired - x) + Kd.dot(v_desired - v)
                tau = np.matmul(M_Jinv, force_impedance - dJ@dq) + h
            else:
                tau = np.zeros(NQ) + h
        
        if controller == 2:     # Cartesian Impedance Control (M. Mayr)
            # Nullspace projection using Moore-Penrose pseudoinverse, Eq. 4:
            JTpinv = np.linalg.inv(J @ J.T) @ J # converge
            nan_jinv = np.isnan(JTpinv).any()
            if not nan_jinv:
                # maybe we need IK here (q_des)
                nullspace_compliance = Kns.dot(q_desired - q) - Dns.dot(dq)
                tau_ns  = (np.eye(NQ) - J.T @ JTpinv) @ nullspace_compliance
                # x_desired must be consistent with q_desired (IK)...
                force_impedance = Kp.dot(x_desired - x) + Kd.dot(v_desired - v)
                tau     = J.T @ force_impedance + tau_ns + h
            else:
                tau = np.zeros(NQ) + h
        
        if controller == 3:     # Cartesian Impedance with TS dynamics (Eq. 3.14 without inertia shaping... maybe we should do Eq. 3.18)
            # Cartesian Impedance
            force_impedance = Kp.dot(x_desired - x) + Kd.dot(v_desired - v)
            # Compute Lambda and u avoiding J^-1:
            A = J @ np.linalg.inv(M)
            Lambda = np.linalg.inv(A @ J.T)
            # Check this math...
            A_inv = np.linalg.pinv(A)
            nan_jinv = np.isnan(A_inv).any()
            #u_js   = h - g - A_inv @ dJ @ dq 
            #tau = g + u_js + J.T @ (force_impedance + Lambda.dot(a_desired))
            tau = h - A_inv @ dJ @ dq + J.T @ (force_impedance + Lambda.dot(a_desired))
            # TODO: Nullspace projection

        if controller == 4:
            # Cartesian Impedance with a_d = v_d = 0 
            nan_jinv = np.linalg.det(J6) == 0
            v = J6 @ dq
            pos_err =  x_desired - x
            rpy_err = pin.rpy.matrixToRpy(Ree_des @ Ree.T)
            err = np.concatenate([pos_err, rpy_err])
            force_impedance = Kp.dot(err) + Kd.dot(-v)
            # Check that this expression only compensate for the JS dynamics, not the TS dynamics
            # Even so, if det(J) == 0 the controller becomes instable...
            tau = g + J6.T @ force_impedance
        
        #JTpinv = np.linalg.inv(J @ J.T) @ J # converge
        check_jinv.append(nan_jinv)
        
        #tau[0] = 0.5 * math.sin( 2 * pie * 0.4 * sim_time)
        #tau = tau + h -0.2*dq

        # Disturbance:
        if sim_time > disturbance_t:
            force_ext[0] = -fe_amp * math.sin(fe_angf * sim_time)
            tau_ext = J.T @ force_ext
        
        ## Simulate Dynamics ##
        # Forward Dynamics (Articulated-Body Algorithm):
        ddq = pin.aba(robot.model, robot.data, q, dq, tau + tau_ext)
        # Integration:
        dq = dq + dt * ddq
        q  = pin.integrate(robot.model, q, dt*dq)

        #viz.sleep(1/60) # reduce the visualization fps ... check 'pip install loop_rate_limiters'
        viz.display(q)

        m_kin = math.sqrt(np.linalg.det(J6.dot(J6.T)))
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