

import cv2
import time
import os
import numpy as np
import math
from manipulate_img import *
from get_img import *
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



def find_aruco_rotation(res_img):
    global dick
    c=[]
    gray = cv2.cvtColor(res_img, cv2.COLOR_BGR2GRAY)
    res = cv2.aruco.detectMarkers(gray, dick)

    marker=17
    
    if res[1] is not None and marker in res[1]: 
    
        index=np.where(res[1]==marker)[0][0]
        
        x1, y1 = int(res[0][index][0][0][0]), int(res[0][index][0][0][1])
        x2, y2 = int(res[0][index][0][1][0]), int(res[0][index][0][1][1])
        dx = x2 - x1
        dy = y2 - y1
        
       
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        return angle_deg
    else:
        return None


    

def create_picture():
    window = np.zeros((500,1000,3), dtype='uint8')
    cv2.circle(window, (500,500), 100, (125, 125, 0), thickness=-11)
    cv2.circle(window, (500,500), 500, (20, 75, 9), thickness=1)
    #cv2.rectangle(window,(500-143), 100, (100, 300), (255,255,255), 5)
    return window


def field_visualisation(field,x,y):
    field = cv2.resize(field,(640, 480))
    cv2.circle(field, (x,y), 10, (0,0,255), thickness=-1)
    cv2.imshow('field',field)




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
    #print('dafk................',dafk[0][0],'!!!!!!!!')
    try:
        cv2.line(res_img, (0,dafk[0][0]), (300,dafk[0][0]), (0,0,255), 9)
        cv2.line(res_img, (0,dafk[0][len(dafk[0])-1]), (300,dafk[0][len(dafk[0])-1]), (0,0,255), 9)
    except:
        print("robot is currently not in ChargingZone Sector 1")
    find_aruco_rotation(res_img)
    cv2.imshow('warp',res_img)
    print('res img shape', res_img.shape)
    angle = find_aruco_rotation(res_img)
    if angle is not None:
        if 0<=abs(angle)<=90:
            print('angle: ',abs(angle))
        else:
            print('angle: ',abs(angle)-90)
    else: print('cannot see aruco on roobt')
    
    
        
    print("..........................................................")
    #field_visualisation(field,,)################

    
    cv2.imshow("img", frame)
    nigga = create_picture()
    cv2.imshow('nigga', nigga)
    key = cv2.waitKey(100) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
