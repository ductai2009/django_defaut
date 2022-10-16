import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import imutils
import random
import string
from numpy import asarray
from PIL import Image, ImageDraw, ImageOps
from PIL.ImageFilter import (
   BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
   EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN,MedianFilter
)



def xoayAnh(img, do):
    rotated = img.rotate(do)
    return rotated
def cut_tron(img):
    npImage=np.array(img)
    h,w=img.size
    alpha = Image.new('L', img.size,0)
    draw = ImageDraw.Draw(alpha)
    draw.pieslice([0,0,h,w],0,360,fill=255)
    npAlpha=np.array(alpha)
    npImage=np.dstack((npImage,npAlpha))
    final_img_arr = Image.fromarray(npImage)
    return final_img_arr



def chinhAnh(img_nho, img_lon, angle):
    img_nho =  xoayAnh(img_nho, -angle)
    img_lon =  xoayAnh(img_lon, angle)
    ig = cut_tron(img_nho)   
    img_lon.paste(im=ig, box=(68,68), mask = ig)
    img_lon = img_lon.filter(MedianFilter(size = 3))
    dVien = img_lon.filter(FIND_EDGES)
    return dVien, img_lon


def capChaTronTikTok(small_img, big_img):
    score = 9999999999999
    do = 0
    a = np.zeros([347,347])
    b = cv2.circle(a, (173,173), color=1, radius=107, thickness=3)
    for i in range(1, 180):
        dVien,_ = chinhAnh(small_img, big_img, i)
        # dVien= asarray(dVien)
        dVien = ImageOps.grayscale(dVien)
        matNa = dVien * b
        # matNa= asarray(matNa)
        tong = matNa.sum()
        if int(tong) < score:
            # print(tong)
            # print(i)
            score = tong
            do = i
    _, img = chinhAnh(small_img, big_img, do)
    return img

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str