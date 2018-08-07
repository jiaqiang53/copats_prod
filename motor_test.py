# encoding:utf-8
import cv2
import sys
import cmath
import serial
import string
import binascii
import time


def sign(x):
    if x >= 0:
        return 0
    else:
        return 1


def x_y_to_com(angle_x,angle_y):
    a = [0x73]
    a.append(int(sign(angle_x)))
    a.append(int(abs(angle_x) // 256))
    a.append(int(abs(angle_x) % 256))
    a.append(int(sign(angle_y)))
    a.append(int(abs(angle_y) // 256))
    a.append(int(abs(angle_y) % 256))
    a.append(0x65)
    return a


def run():
    port = serial.Serial('/dev/ttyUSB0', 115200)
    # port1 = serial.Serial('/dev/ttyUSB1', 115200)
    x_angle = 50
    y_angle = 5
    cnt = 0
    # x_angle = round(50000.0 / (3 * x_angle * 20000 * 2.0 / 360.0), 4)
    # y_angle = round(50000.0 / (1 * y_angle * 20000 * 2.0 / 360.0), 4)

    while True:
        cnt += 1
        if cnt % 20 == 0:
            cnt = 0
            x_angle = -x_angle
            y_angle = -y_angle

        time.sleep(0.1)
        res_angle_x = 'x' + str(x_angle) + 'n'
        res_angle_y = 'y' + str(y_angle) + 'n'

        pulses_per_circle = 8000.0
        angle_x = x_angle * pulses_per_circle / 360.0
        angle_y = y_angle * pulses_per_circle / 360.0

        # print hex(int(angle_x))

        print '73%02X%04X%02X%04X65' % (sign(angle_x), abs(angle_x), sign(angle_y), abs(angle_y))
        command = x_y_to_com(angle_x,angle_y)
        # d = bytes.fromhex(command)
        # print(command)
        print "cnt:", cnt, "x_angle: ", res_angle_x, "y_angle: ", res_angle_y
        if port.is_open:
            port.flushInput()
            print(command)
            angle_resx = port.write(command)
        # if port1.is_open:
        #     port1.flushInput()
        #     angle_resy = port1.write(res_angle_y)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break


if __name__ == "__main__":
    run()