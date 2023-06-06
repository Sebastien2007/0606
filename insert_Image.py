import cv2
import cv2.aruco as aruco

def create_aruco_marker(id, dict_id=aruco.DICT_4X4_50, size=50,border_size=10):
    aruco_dict = aruco.Dictionary_get(dict_id)
    img = aruco.drawMarker(aruco_dict, id, size)
    img = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, None, value=255)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

import cv2
import numpy as np

def insert_rotated_image(big_image, small_image, angle, center):
    if big_image.shape[2] == 3:
        big_image = cv2.cvtColor(big_image, cv2.COLOR_BGR2BGRA)
    if small_image.shape[2] == 3:
        small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2BGRA)

    (h, w) = small_image.shape[:2]
    center_img = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center_img, angle, 1.0)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    rotation_matrix[0, 2] += (new_w / 2) - center_img[0]
    rotation_matrix[1, 2] += (new_h / 2) - center_img[1]

    small_image = cv2.warpAffine(small_image, rotation_matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    y, x = center
    y1, y2 = max(0, y - new_h // 2), min(big_image.shape[0], y + new_h // 2)
    x1, x2 = max(0, x - new_w // 2), min(big_image.shape[1], x + new_w // 2)

    alpha_s = small_image[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        big_image[y1:y2, x1:x2, c] = (alpha_s * small_image[:, :, c] +
                                      alpha_l * big_image[y1:y2, x1:x2, c])
    return big_image

aruco = create_aruco_marker(17)
bg_image = cv2.imread("field.jpg", cv2.IMREAD_UNCHANGED)
bg_image = cv2.resize(bg_image, (580,380), interpolation = cv2.INTER_AREA)

new_img = insert_rotated_image(bg_image, aruco, 2, (300,200))

cv2.imshow("aruco", new_img)
