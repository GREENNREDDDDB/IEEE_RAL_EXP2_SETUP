import sys

# sys.path.append("../lib")
sys.path.append("/home/njau/robotic_arm/z1_sdk/lib")
import unitree_arm_interface
import numpy as np
import time

print("Press ctrl+\ to quit process.")

np.set_printoptions(precision=3, suppress=True)
arm = unitree_arm_interface.ArmInterface(hasGripper=True)
armState = unitree_arm_interface.ArmFSMState

arm.loopOn()
arm.labelRun("sa_start")

while True:
    arm.labelRun("sa_a1")
    time.sleep(2)
    arm.labelRun("sa_a2")
    time.sleep(2)
    arm.labelRun("sa_a3")
    time.sleep(2)
    run = input("still want to continue?(y/n):")
    if run == "n":
        arm.labelRun("sa_start")
        break

# ser.close()
arm.backToStart()
arm.loopOff()
