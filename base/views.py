
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import Room, Topic, Mesenge, User, image_64
from .forms import RoomForm, UserForm, CreateUser,UpLoadImg
from .modelML import predict_img
from django.db.models import Q
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.forms import UserCreationForm
import keras
import base64
from PIL import Image
import numpy as np
import os
import cv2
import random
from tensorflow.keras.models import load_model
import tensorflow as tf
from django.http import FileResponse
from PIL import Image





# model = load_model('model1.h5')
# decode = {'living room': 0,
#  'lion': 1,
#  'smiling dog': 2,
#  'lion with closed eyes.': 3,
#  'dog': 4,
#  'dog with a collar on its neck': 5,
#  'horse': 6,
#  'canine.': 7,
#  'parrot': 8,
#  'lion with an open mouth.': 9,
#  'conference room': 10,
#  'giraffe': 11,
#  'bedroom': 12,
#  'boat': 13,
#  'bird.': 14,
#  'lion with open eyes.': 15,
#  'lion with mane on its neck.': 16,
#  ' elephant': 17,
#  'car': 18,
#  'seaplane': 19,
#  ' airplane': 20,
#  'truck': 21}

# def name_to_vec(name:str):
#     if name in decode.keys():
#             return tf.keras.utils.to_categorical(decode[name] , len(decode))
#     else:
#         return np.zeros(len(decode))
# def cut_img(img):
    
#   num = 9
#   w, h = img.size
#   range_pixel = int(w/3)
#   list_img = []
#   for i in range(int(num**0.5)):
#     for j in range(int(num**0.5)):
#       list_img.append(img[i*range_pixel:i*range_pixel+range_pixel,j*range_pixel:j*range_pixel+range_pixel])
#   return list_img

# def predict_img(imgfile):
#     # img_path = "./templates/image/" + imgfile.filename
#     # imgfile.save(img_path)
    
#     # img_9 = cv2.imread(img_path)[:,:,::-1]
#     # print(img_9.shape)
#     # name = img_path.split('_')[-1].split('.')[0]
#     # img_9 = cv2.imread(img_9)
#     path = 'static/images/img/'
#     path  = 'static' + imgfile
#     name = 'dog'
#     img_9 = Image.open(path)
#     # print('du doannnnnnnn', img_9.shape[0])
#     img_9 = img_9.resize((384,384))
    
#     list_img = cut_img(img_9)
#     x1 = np.array(list_img)
#     x2 = np.array([name_to_vec(name)]*9)
#     model.predict([x1,x2])*1
#     predict = (model.predict([x1,x2]) >= 0.5).reshape(1,9)*1
#     predict = str(predict)
#     return predict


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
    
# model = load_model('model1.h5')
# Create your views here.
def logOut(request):
    logout(request)
    return redirect('login')

def signUp(request):
    form = CreateUser()
    if request.method == 'POST':
        form = CreateUser(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            # user.username = user.username.lower()
            user.save()
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, 'Bạn đã sai ở đâu đó!')
    context = {'form': form}
    return render(request, 'base/login.html', context)

def logIn(request):
    page = 'login_page'
    if request.user.is_authenticated:
        return redirect('home')
    if request.method == 'POST':
        username = request.POST.get('username').lower()
        password = request.POST.get('password')
        try:
            user = User.objects.get(username = username)
        except:
            messages.error(request, 'Username này không tồn tại!')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, 'Lỗi rồi!')
    context = {'page': page}
    # context = {'user': user}
    return render(request, 'base/login.html', context)

def home(request):
    # if request.method == 'GET': 
    #     qq = request.GET.get('qq') if request.GET.get('qq') != None else ''
    #     if qq != '':
    #         rooms = Room.objects.filter(name__icontains = qq)
    #         room_count = rooms.count()
    #     q = request.GET.get('q') if request.GET.get('q') != None else ''
    #     if q != '':
    #         rooms = Room.objects.filter(Q(topic__name__icontains = q) |
    #                                     Q(name__icontains = q)
    #                                     )
    #         room_count = rooms.count()
    q = request.GET.get('q') if request.GET.get('q') != None else ''
    rooms = Room.objects.filter(Q(topic__name__icontains = q) |
                                        Q(name__icontains = q)|
                                        Q(description__icontains=q)
                                        )
    room_count = rooms.count()
    topics = Topic.objects.all()[0:5]
    message_home = Mesenge.objects.filter(Q(room__topic__name__icontains = q))[0:5]
    context = {'rooms' : rooms, 'topics': topics, 'room_count': room_count,'message_home': message_home}
    return render(request, 'base/home.html',context)

def profile(request, pk):
    user = User.objects.get(id=pk)
    rooms = Room.objects.filter(host = user)
    message_home = Mesenge.objects.filter(user = user)
    topics = Topic.objects.filter()
    room_count = rooms.count()
    context = {'user': user, 'rooms': rooms,'room_count':room_count, 'message_home':message_home
               ,'topics':topics}
    return render(request,'base/profile.html' ,context)

@login_required(login_url='login')
def create_room(request):
    form = RoomForm()
    
    
    topics = Topic.objects.all()
    if request.method == 'POST':
        form = RoomForm(request.POST)
        name_topic = request.POST.get('topic')
        topic, created = Topic.objects.get_or_create(name = name_topic)
        Room.objects.create(
            host = request.user,
            topic = topic,
            name = request.POST.get('name'),
            description = request.POST.get('description'),
        )
        return redirect('home')
    context = {'form': form, 'topics':topics}
    return render(request, 'base/create_room.html', context)

@login_required(login_url='login')
def updateRoom(request, pk):
    room = Room.objects.get(id = pk)
    form = RoomForm(instance=room)
    topics = Topic.objects.all()
    if request.user != room.host:
        return HttpResponse("Bạn không có quyền thực hiện thao tác này!!")
    if request.method == 'POST':
        name_topic = request.POST.get('topic')
        topic, created = Topic.objects.get_or_create(name = name_topic)
        form = RoomForm(request.POST, instance=room)
        room.name = request.POST.get('name')
        room.topic = topic
        room.description = request.POST.get('description')
        room.save()
        return redirect('home')
    context = {'form': form, 'topics':topics,'room':room}
    return render(request, 'base/create_room.html', context)

@login_required(login_url='login')
def deteleRoom(request, pk):
    room = Room.objects.get(id = pk)
    if request.method == 'POST':
        room.delete()
        return redirect('home')
    if request.user != room.host:
        return HttpResponse("Bạn không có quyền thực hiện thao tác này!!")
    context = {'room': room}
    return render(request, 'base/delete.html', context)

@login_required(login_url='login')
def deteleMessange(request, pk):
    message = Mesenge.objects.get(id = pk)
    # id_room = pk
    if request.user != message.user:
        return HttpResponse("Bạn không có quyền thực hiện thao tác này!!")
    if request.method == 'POST':
        message.delete()
        return redirect('home')
    context = {'message': message}
    return render(request, 'base/delete.html', context)
def updateUser(request, pk):
    user = User.objects.get(id = pk)
    form = UserForm(instance=user)
    if request.method == 'POST':
        form = UserForm(request.POST, request.FILES,  instance=user)
        if form.is_valid():
            form.save()
            return redirect('profile', pk = user.id)
    context = {'user':user, 'form':form}
    return render(request, 'base/edit-user.html', context)
def room(request, pk):
    code_base64 = 0
    room = Room.objects.get(id = pk)
    size = image_64.objects.all()
    for i in size:
        if i.codeImg != 'null':
            code_base64 = i
    filename = 'Bạn chưa úp ảnh nào lên'
    try:
        imgdata = base64.b64decode(code_base64.codeImg)
        path = 'static/images/img/'
        
        filename =  code_base64.name +'.jpg'
        with open(path + filename, 'wb') as f:
            f.write(imgdata)
    except:
        pass
    room_message = room.mesenge_set.all()
    participants = room.participants.all()
    
    if request.method == 'POST':
        messange = Mesenge.objects.create(
            user = request.user,
            room = room,
            body = request.POST.get('cmt')
        )
        room.participants.add(request.user)
        return redirect('room', pk = room.id)
    # message = Mesenge.objects.filter(room.body)
    
    content = {'room': room, 'room_message': room_message,
               'participants':participants,'imgdata':filename}

    # os.remove(path + filename)
    return render(request, 'base/room.html', content)

def topicPage(request):
    q = request.GET.get('q') if request.GET.get('q') != None else ''
    rooms = Room.objects.filter(Q(topic__name__icontains = q) |
                                        Q(name__icontains = q)|
                                        Q(description__icontains=q)
                                        )
    topic = Topic.objects.filter()
    context = {'rooms': rooms,'topic':topic}
    return render(request, 'base/topics.html', context)

def upImage(request, pk):
    room = Room.objects.get(id= pk)
    room_all = Room.objects.all()
    q = request.GET.get('room')
    if q == None:
        folder_img = image_64.objects.all()
    else:
        folder_img = image_64.objects.filter(room = q)

    # size = image_64.objects.all()
    # for i in size:
    #     if i.codeImg != 'null':
    #         code_base64 = i
    # filename = 'Bạn chưa úp ảnh nào lên'
    # try:
    #     imgdata = base64.b64decode(code_base64.codeImg)
    #     path = 'static/images/img/'
    #     filename =  code_base64.name +'.jpg'
    #     with open(path + filename, 'wb') as f:
    #         f.write(imgdata)
    # except:
    #     pass
    # img64 = image_64.objects.get(id= pk)
    # img1 = 'none'
    # form_upimg = UpLoadImg(request.POST)
    # if request.method == 'POST':
    #     form_upimg = UpLoadImg(request.POST, request.FILES)
    #     if form_upimg.is_valid():
    #         image_64.objects.create(
    #             user = request.user,
    #             room = room,
    #             name = str(random.randrange(1,100)),
    #             # image = form_upimg.save(commit=False),
    #             # image = request.FILES.get(form_upimg)
    #         )
    #         img1 = request.FILES.get('img')
    #         path = 'static/images/img1/'
    #         filename =  str(random.randrange(1,30)) +'.jpg'
    #         with open(path+filename, 'rb') as f:
    #             img1 = [x.decode('utf8').strip() for x in f.readlines()]
    #         # text64 = base64.b64encode(img1)

    #         # img64.user = request.user
    #         # img64.room = room
    #         # form_upimg.save()
    #         return redirect('room', pk=room.id )

    context = {'room': room,'room_all':room_all, 'folder_img': folder_img}
    return render(request, 'base/photo.html', context)


# def show_img():
#     room = 
#     return render(request,'base/show_img.html, ')

def upimg(request, pk):
    room = Room.objects.get(id = pk)
    kp = 'ko'
    form = UpLoadImg(request.POST, request.FILES)
    if request.method == 'POST':
        form = UpLoadImg(request.POST, request.FILES)
    
    if request.method == 'POST':
        form = UpLoadImg(request.POST, request.FILES)
        image_64.objects.create(
            user = request.user,
            room = room,
            name = request.POST.get('nameimg'),
            image = request.FILES.get('images'),
        )
        return redirect('UpLoadImg', room.id)
    context = {'kp':kp}
    return render(request, 'base/upimage.html', context)



def activityUser(request):
    rooms = Room.objects.all()
    messanges_hoom = Mesenge.objects.all()[0:5]
    context = {'rooms':rooms,'messanges_hoom':messanges_hoom}
    return render(request, 'base/activityUser.html', context)