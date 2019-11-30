import numpy as np
import cv2
import glob
from scipy import linalg as LA
from scipy.ndimage import maximum_filter

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
tau = 50000
image_list = []
path = "C:\Data\Software\hotelImages\*.*"
for file in glob.glob(path):
    a= cv2.imread(file)
    image_list.append(a)
print("Number of images")
print(len(image_list))
currImage = image_list[0]

def getSobel(axis):
    if axis == 0:
        return np.array([ [-1,-2,-1],
                        [ 0, 0, 0],
                        [ 1, 2, 1]])
    else:
        return np.array([ [-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

def detect_corner(img, k):
    
    img1 = img.copy()
    sobel_x = getSobel(0)
    sobel_y = getSobel(1)

    img = np.float64(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))

    Ix = cv2.filter2D(img, -1, sobel_x)
    Iy = cv2.filter2D(img, -1, sobel_y)

    Ixx = Ix ** 2
    Ixy = Ix * Iy
    Iyy = Iy ** 2

    gaussian = cv2.getGaussianKernel(5,0.5)
    Ixx = cv2.filter2D(Ixx, -1, gaussian)
    Ixy = cv2.filter2D(Ixy, -1, gaussian)
    Iyy = cv2.filter2D(Iyy, -1, gaussian)
    row, col = img.shape
    b = Ixx*Iyy - np.square(Ixy) - (k*np.square(Ixx + Iyy))
    cv2.imshow('b', b)
    c, d = show_corners(b, img1, tau)
    return c, d

def show_corners(corners, img, threshold):
    f = 0
    img1 = img.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = img_rgb.copy()
    corner = non_maximum_suppresor(corners, 1, 5)
    corner = cv2.dilate(corner, None)
    cornerList = []
    print(np.max(corners))
    img_gray[corners > threshold ]= 255
    img1[corners > threshold*np.max(corners)] = [0,255,0]
    for x in range(img_gray.shape[0]):
        for y in range(img_gray.shape[1]):
            if img_gray[x,y] == 255:
                cv2.circle(img1, (y,x), 2, (0,255,0), 1)
                f = f + 1
    print("Number of corners detected")
    print(f)
    a = np.where(img_gray == 255)
    cv2.imshow('corners', img_gray)
    cv2.imshow('CORNERS', img1)
    cv2.imwrite("corners.png", img1)
    return img_gray, np.array(a)

def non_maximum_suppresor(image, stepSize, windowSize):
    
    for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                w = x + windowSize
                h = y + windowSize
                h = np.min([h,image.shape[0]])
                w = np.min([w,image.shape[1]])
                window = image[y:h, x:w]
                i = np.argmax(window)
                mask = np.zeros((h-y,w-x))
                mask[int(i/mask.shape[1])][int(i%mask.shape[1])] = 1
                image[y:h, x:w] = np.multiply(window, mask)

    return image

a, b = detect_corner(currImage, 0.06)
print(b)
