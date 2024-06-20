from roboticstoolbox import DHRobot, RevoluteDH
from roboticstoolbox.tools import plot

# 定义关节参数，包括关节的旋转轴和DH参数
L1 = RevoluteDH(d=0.065)
L2 = RevoluteDH(d=0.1115)
L3 = RevoluteDH(a=-0.35, d=0.1115)
L4 = RevoluteDH(a=-0.132, d=0.1685)
L5 = RevoluteDH(a=-0.06, d=0.1685)
L6 = RevoluteDH(a=-0.0128, d=0.1685)

# 创建机械臂模型
robot = DHRobot([L1, L2, L3, L4, L5, L6])
print(robot)
# 设置关节坐标
q = [0.5, 0.5, 0.5, 0, 0.5, 0.5]  # 关节角度，单位为弧度
# robot.q = q
# # 打印机械臂的末端位姿
print(robot.fkine(q))
# # 可以通过设置关节角度来计算机械臂的末端位姿
# q = [0, 0, 0, 0, 0, 0.5]  # 设置关节角度
# print(robot.fkine(q))
robot.q
# 将机械臂模型显示出来
robot.plot(q, block=True)