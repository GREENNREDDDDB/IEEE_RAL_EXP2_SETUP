import roboticstoolbox as rtb
from spatialmath import SE3

robot = rtb.models.Panda()
Tep = SE3.Trans(0.6, -0.3, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
sol = robot.ik_LM(Tep)         # solve IK
print(sol)

q_pickup = sol[0]
Res = robot.fkine(q_pickup)
print(Res)    # FK shows that desired end-effector pose was achieved

qi = [1, 0, 0, 0, -1, 0, 1]
# qt = rtb.jtraj(robot.qr, q_pickup, 50)

# Compute a joint-space trajectory from initial joint coordinate to final joint coordinate
qt = rtb.jtraj(qi, q_pickup, 50)

robot.plot(qi, backend='swift', movie='panda1.gif', block=True)
