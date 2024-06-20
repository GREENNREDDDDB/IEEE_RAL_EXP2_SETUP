import sys

# sys.path.append("../lib")
sys.path.append("/home/njau/robotic_arm/z1_sdk/lib")
import unitree_arm_interface
import numpy as np
import open3d as o3d
from drive_D435.get_pcd import get_point_cloud
from drive_D435.get_realtime import get_real_time
from kinematics.computeTransform import transform_matrix
from plant.cloudTransform import transform

print("Press ctrl+\ to quit process.")

np.set_printoptions(precision=3, suppress=True)
arm = unitree_arm_interface.ArmInterface(hasGripper=True)
armState = unitree_arm_interface.ArmFSMState

arm.loopOn()
arm.labelRun("sa_start")

arm.labelRun("sa_a1")
cloud1 = get_point_cloud("sa1")

arm.labelRun("sa_a2")
cloud2 = get_point_cloud("sa2")
arm.labelRun("sa_a3")
cloud2 = get_point_cloud("sa3")
arm.labelRun("sa_a4")
cloud2 = get_point_cloud("sa4")
arm.labelRun("sa_a5")
cloud2 = get_point_cloud("sa5")
arm.labelRun("sa_a6")
cloud2 = get_point_cloud("sa6")
arm.labelRun("sa_a7")
cloud2 = get_point_cloud("sa7")
arm.labelRun("sa_a8")
cloud2 = get_point_cloud("sa8")

# tr_matrix = transform_matrix()
# transform(cloud2, cloud1, tr_matrix)
# get_real_time()

# while True:
#     arm.labelRun("sa_a1")
#     time.sleep(2)
#     arm.labelRun("sa_a2")
#     time.sleep(2)
#     arm.labelRun("sa_a3")
#     time.sleep(2)
#     run = input("still want to continue?(y/n):")
#     if run == "n":
#         arm.labelRun("sa_start")
#         break

# ser.close()
arm.backToStart()
arm.loopOff()
