from roboticstoolbox import ET, ERobot
import roboticstoolbox as rtb
from spatialmath import SE3
import numpy as np

# Puma dimensions (m), see RVC2 Fig. 7.4 for details
l1 = 0.065; l2 = 0.0465; l3 = -0.35; l4 = 0.218; l5 = 0.0575; l6 = 0.072; l7 = 0.0472
e = ET.tz(l1) * ET.Rz() \
    * ET.tz(l2) * ET.Ry() \
    * ET.tx(l3) * ET.Ry() \
    * ET.tx(l4) * ET.tz(l5) * ET.Ry() \
    * ET.tx(l6) * ET.Rz() \
    * ET.tx(l7) * ET.Rx()

# print(e)
robot = rtb.Robot(e)

# Inverse kinematics
Tep = SE3.Trans(0.3, 0.3, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
Res = robot.ik_LM(Tep)
# Final joint coordinate
qf = Res[0]
print(qf)
# robot.fkine(q[0])

# Define the initial joint coordinate
# qi = np.array([0, 1.57, -1, 0, 0, 0])
qi = np.array([0, 0, 0, 0, 0, 0])
sa_a1 = np.array([-0.004756, 0.478025, -0.551739, -0.721797, 0.000183, -0.092264])
# Compute a joint-space trajectory from initial joint coordinate to final joint coordinate
qt = rtb.jtraj(qi, qf, 50)

# Plot the animation of trajectory
# robot.plot(qt.q, backend='pyplot', movie='panda1.gif')

# Plot the pose
robot.plot(sa_a1, backend='pyplot', block=True)
print(robot)
