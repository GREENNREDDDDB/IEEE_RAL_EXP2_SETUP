# coding:utf-8

import serial
import time

ser = serial.Serial('/dev/ttyUSB1',115200,timeout=1,bytesize=8, stopbits=1)
grasp_string = "AABB0001CCDD"
drop_string = "AABB0000CCDD"
grasp = bytes(bytearray.fromhex(grasp_string))
drop = bytes(bytearray.fromhex(drop_string))
# lst = [0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47]
# lst_bytes = bytes(lst)
# print(lst_bytes)
ser.close()
ser.open()
# while True:
print(grasp)
# ser.write(lst_bytes)
ser.write(drop)
# time.sleep(0.5)
# read = ser.read(7)
# print(read)
