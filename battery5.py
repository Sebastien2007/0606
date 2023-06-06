import socket
import pickle
import struct
import cv2
import time
import os
import numpy as np
import math
dick = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
field = cv2.imread('field.jpg')
h_min=(0,140,173)
h_max=(255,255,255)

numba=0
save_h = np.array([])
use=0

def crop_img(img):
    cropped_image = img[0:480, 90:550]
    return cropped_image
    
with open('param.txt') as f:
    K = eval(f.readline())
    D = eval(f.readline())

def undistort(img):
    DIM = img.shape[:2][::-1]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img[::]


def get_image():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_ip = '192.168.0.222'
    port = 9988
    client_socket.connect((host_ip, port))

    data = b""
    payload_size = struct.calcsize("Q")
    while True:
        while len(data) < payload_size:
            packet = client_socket.recv(4*1024)
            if not packet: break
            data += packet
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            data += client_socket.recv(4*1024)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data)
        client_socket.close()
        return frame
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


    
def warp_da(frame):
    global dick
    c=[]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.aruco.detectMarkers(gray, dick)
    if res[1] is not None and (0 in res[1]) and (1 in res[1])and (2 in res[1]) and (3 in res[1]):
        for i in range(4):
            
            marker=i
            index=np.where(res[1]==marker)[0][0]
            pt0=res[0][index][0][marker].astype(np.int16)
            c.append(list(pt0))
            cv2.circle(frame, pt0, 10, (0,0,255), thickness=-1)
            
        h, w, _ = frame.shape
        input_pt=np.array(c)
        output_pt=np.array([[0,0], [w,0],[w,h],[0,h]])
        h_, _ = cv2.findHomography(input_pt, output_pt)
        np.savetxt('h_wh.txt', h_)
       

        return h_
    else:
        return np.loadtxt('h_wh.txt')

def create_picture():
    window = np.zeros((500,1000,3), dtype='uint8')
    cv2.circle(window, (500,500), 100, (125, 125, 0), thickness=-11)
    cv2.circle(window, (500,500), 500, (20, 75, 9), thickness=1)
    #cv2.rectangle(window,(500-143), 100, (100, 300), (255,255,255), 5)
    return window


def field_visualisation(field):
    field = cv2.resize(field,(640, 480))
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
    field_visualisation(field)

    
    cv2.imshow("img", frame)
    nigga = create_picture()
    cv2.imshow('nigga', nigga)
    key = cv2.waitKey(100) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
