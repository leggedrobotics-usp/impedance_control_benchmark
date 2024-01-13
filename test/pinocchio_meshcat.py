#   Test Pinocchio execution
#   Leonardo F. dos Santos, 2023 | qleonardolp
#   Run on interactive mode: $ python3 -i pinocchio_meshcat.py
import pinocchio as pin
import numpy as np
import time as tm
import math
from math import pi as pie


from pinocchio.visualize import MeshcatVisualizer
from example_robot_data import load

robot_name = "ur5"
robot = load(robot_name)
viz = MeshcatVisualizer()

robot.setVisualizer(viz)
robot.initViewer(open=True)
robot.loadViewerModel()

NQ, NV = robot.model.nq, robot.model.nv

q = np.array([0.00, -1.00, 1.00, -pie, -pie / 2, 0])
q = pin.neutral(robot.model)
dq = np.zeros(NV)

delta_t = []
avg_freq = 0

sim_duration = 10  # [s]
sim_dt = 0.020  # [s]
sim_steps = int(sim_duration / sim_dt)

try:
    print("Simulation started. Press CTRL+C to stop.")
    tic = tm.time()
    for k in range(sim_steps):
        dt = sim_dt
        sim_time = k * dt
        tau = np.zeros(NQ)
        tau[5] = 0.5 * math.sin(2 * pie * 0.2 * sim_time)
        pin.computeAllTerms(robot.model, robot.data, q, dq)
        h = robot.gravity(q)
        tau_ref = tau * 0 + h
        # tau_ref = tau + h
        # Forward Dynamics (Articulated-Body Algorithm):
        ddq = pin.aba(robot.model, robot.data, q, dq, tau_ref)
        # Integration:
        dq = dq + dt * ddq
        q = pin.integrate(robot.model, q, dt * dq)
        viz.display(q)
    ellapsed_time = tm.time() - tic
    print(
        f"\nSimulation ended. ({ellapsed_time:.2f} s)\n"
    )  # :.2f formatting float with 2 decimals

except KeyboardInterrupt:
    # Stop loop with CTRL+C
    exit
