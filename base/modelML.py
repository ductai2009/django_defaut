import keras
from .models import Room, Topic, Mesenge, User, image_64
import base64
from PIL import Image
import numpy as np
import os
import cv2
import random
from django.shortcuts import render, redirect
from tensorflow.keras.models import load_model
import tensorflow as tf
from django.http import FileResponse
from PIL import Image

model = load_model('model1.h5')
decode = {'living room': 0,
 'lion': 1,
 'smiling dog': 2,
 'lion with closed eyes.': 3,
 'dog': 4,
 'dog with a collar on its neck': 5,
 'horse': 6,
 'canine.': 7,
 'parrot': 8,
 'lion with an open mouth.': 9,
 'conference room': 10,
 'giraffe': 11,
 'bedroom': 12,
 'boat': 13,
 'bird.': 14,
 'lion with open eyes.': 15,
 'lion with mane on its neck.': 16,
 ' elephant': 17,
 'car': 18,
 'seaplane': 19,
 ' airplane': 20,
 'truck': 21}

def name_to_vec(name:str):
    if name in decode.keys():
            return tf.keras.utils.to_categorical(decode[name] , len(decode))
    else:
        return np.zeros(len(decode))
def cut_img(img):
    
  num = 9
  w, h = img.size
  range_pixel = int(w/3)
  list_img = []
  for i in range(int(num**0.5)):
    for j in range(int(num**0.5)):
      list_img.append(img[i*range_pixel:i*range_pixel+range_pixel,j*range_pixel:j*range_pixel+range_pixel])
  return list_img

def predict_img(imgfile):
    # img_path = "./templates/image/" + imgfile.filename
    # imgfile.save(img_path)
    
    # img_9 = cv2.imread(img_path)[:,:,::-1]
    # print(img_9.shape)
    # name = img_path.split('_')[-1].split('.')[0]
    # img_9 = cv2.imread(img_9)
    path = 'static/images/img/'
    path  = 'static' + imgfile
    name = 'dog'
    img_9 = Image.open(path)
    # img_9 = cv2.imread(path)
    # img_9 = np.array(img_9)
    # print('du doannnnnnnn', img_9.shape[0])
    img_9 = img_9.resize((384,384))
    
    list_img = cut_img(img_9)
    x1 = np.array(list_img)
    x2 = np.array([name_to_vec(name)]*9)
    model.predict([x1,x2])*1
    predict = (model.predict([x1,x2]) >= 0.5).reshape(1,9)*1
    predict = str(predict)
    return predict


def predict(request, pk):
    img = image_64.objects.get(id = pk)
    # path = 'static/images/img/'
    # img9 = img
    # path = 'static/images/img/'
    # filename =  str(img.image)
    
    # image_data = open(path + filename, "rb").read()
    # image = FileResponse(image_data)

    kp = predict_img(img.image.url)
    # kp = img.image.path
    return render(request, 'base/predict.html', {'kp': kp})
    