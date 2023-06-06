import cv2
import time
import os
import numpy as np
import math
from manipulate_img import *
from get_img import *
from aruco_angle import *
#from config import *

dick = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
field = cv2.imread('field.jpg')
h_min=(0,140,173)
h_max=(255,255,255)


numba=0
save_h = np.array([])
use=0


with open('param.txt') as f:
    K = eval(f.readline())
    D = eval(f.readline())

while True:
    frame = get_image()
    frame = undistort(frame)
    
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    
    
    h_ = warp_da(frame)
    h, w, _ = frame.shape
    res_img = cv2.warpPerspective(frame, h_, (w,h))

    img_bin = cv2.inRange(res_img[0:480, 90:300], h_min, h_max)
    bitwiseNot = cv2.bitwise_not(img_bin)

    cv2.imshow('ChargingZoneSector1', img_bin)
    cv2.imshow('bitwiseNot', bitwiseNot)

    summa = np.sum(bitwiseNot, axis=1)
    print(summa)
    dafk = np.where(summa>10000)
    
    try:
        cv2.line(res_img, (0,dafk[0][0]), (300,dafk[0][0]), (0,0,255), 9)
        cv2.line(res_img, (0,dafk[0][len(dafk[0])-1]), (300,dafk[0][len(dafk[0])-1]), (0,0,255), 9)
        find_aruco_rotation(res_img)
    except:
        print("robot is currently not in ChargingZone Sector 1")
    
    cv2.imshow('warp',res_img)
    print('res img shape', res_img.shape)
    angle, dot_x, dot_y = find_aruco_rotation(res_img)
    #c.append(list(pt0))
    #    cv2.circle(frame, pt0, 10, (0,0,255), thickness=-1)
    if angle is not None:
        if 0<=abs(angle)<=90:
            print('angle: ',abs(angle))
        else:
            print('angle: ',abs(angle)-90)
    else: print('cannot see aruco on roobt')
    
    
        
    print("..........................................................")
    #field_visualisation
    field_copy = cv2.resize(field,(640, 480))
    try:
        cv2.circle(field_copy, (dot_x, dot_y), 10, (0,0,255), thickness=-1)
    except:
        pass
    cv2.imshow("field", field_copy)
    cv2.imshow("img", frame)
    nigga = create_picture()
    cv2.imshow('nigga', nigga)
    key = cv2.waitKey(100) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
