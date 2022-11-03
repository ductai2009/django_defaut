
from multiprocessing import managers
from django.shortcuts import render, redirect
from rest_framework.decorators import api_view
from django.http import HttpResponse
from .models import *
from .forms import *
from .api import *
from .funsion_base import *
from .webCam import Webcam
from .thead import *
from django.http import StreamingHttpResponse
from PIL import Image as im
from datetime import datetime, timedelta, timezone
import urllib
import time
# from Yolov5_DeepSort_Pytorch.yolov5 import torch, yolov5
import torch, yolov5
from yolov5.utils.general import (check_img_size, non_max_suppression, scale_coords, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from .modelML import *
# from Yolov5_DeepSort_Pytorch.track import pre
from django.db.models import Q
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.forms import UserCreationForm
# import tensorflow.keras as keras
import base64
from PIL import Image
import numpy as np
import os
import io
import numpy as np
import cv2
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import array_to_img
import tensorflow as tf
from django.http import FileResponse
from threading import Thread
from io import BytesIO




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

@login_required(login_url='login')
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
        # form = CreateUser(request.POST)
        # if form.is_valid():
        #     user = form.save(commit=False)
            # user.username = user.username.lower()
            # user.save()
            # login(request, user)
            # return redirect('home')
            
        # else:
        #     messages.error(request, 'Bạn đã sai ở đâu đó!')
        return HttpResponse("Chức năng tạm thời ngưng!")
    context = {'form': form}
    return render(request, 'base/logIn_new.html', context)


def logIn(request):
    page = 'login_page'
    if request.user.is_authenticated:
        return redirect('home')
    if request.method == 'POST':
        username = request.POST.get('email').lower()
        password = request.POST.get('password')
        try:
            user = User.objects.get(username = username)
        except:
            # messages.error(request, 'Rất... xin chàooooo!')
            pass
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, 'Lỗi rồi!')
    context = {'page': page}
    # context = {'user': user}
    return render(request, 'base/logIn_new.html', context)


@login_required(login_url='login')
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


@login_required(login_url='login')
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
    if request.user.username == "ductai":
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
    else:
        return HttpResponse("Chức năng này chỉ có admin được phép thao tác! sr very much!")
    context = {'form': form, 'topics':topics}
    return render(request, 'base/create_room.html', context)

@login_required(login_url='login')
def updateRoom(request, pk):
    room = Room.objects.get(id = pk)
    form = RoomForm(instance=room)
    topics = Topic.objects.all()
    if request.user != "ductai":
        return HttpResponse("Chỉ admin mới có quyền thực hiện thao tác này!!")
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

    if request.user != "ductai":
        return HttpResponse("Bạn không có quyền thực hiện thao tác này!!")
    elif request.method == 'POST':
        room.delete()
        return redirect('home')
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

@login_required(login_url='login')
def updateUser(request, pk):
    user = User.objects.get(id = pk)
    form = UserForm(instance=user)
    if request.user.username != user.username:
        return HttpResponse("Bạn không có quyền thực hiện thao tác này!!")
    elif request.method == 'POST':
        form = UserForm(request.POST, request.FILES,  instance=user)
        if form.is_valid():
            form.save()
            return redirect('profile', pk = user.id)
    context = {'user':user, 'form':form}
    return render(request, 'base/edit-user.html', context)

@login_required(login_url='login')
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


@login_required(login_url='login')
def topicPage(request):
    q = request.GET.get('q') if request.GET.get('q') != None else ''
    rooms = Room.objects.filter(Q(topic__name__icontains = q) |
                                        Q(name__icontains = q)|
                                        Q(description__icontains=q)
                                        )
    topic = Topic.objects.filter()
    context = {'rooms': rooms,'topic':topic}
    return render(request, 'base/topics.html', context)


@login_required(login_url='login')
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
@login_required(login_url='login')
def upimg(request, pk):
    room = Room.objects.get(id = pk)
    form = UpLoadImg(request.POST, request.FILES)
    if request.method == 'POST':
        form = UpLoadImg(request.POST, request.FILES)

    if request.method == 'POST':
        form = UpLoadImg(request.POST, request.FILES)
        code = base64.b64encode(request.FILES.get('images').read())
        code = code.decode('utf-8')
        image_64.objects.create(
            user = request.user,
            room = room,
            name = request.POST.get('nameimg'),
            image = request.FILES.get('images'),
            codeImg = code,
        )
        return redirect('UpLoadImg',pk = room.id)
    return render(request, 'base/upimage.html')


@login_required(login_url='login')
def activityUser(request):
    rooms = Room.objects.all()
    messanges_hoom = Mesenge.objects.all()[0:5]
    context = {'rooms':rooms,'messanges_hoom':messanges_hoom}
    return render(request, 'base/activityUser.html', context)

print(torch.cuda.is_available())
#load model
# model = yolov5.load('yolov5s.pt')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
device = select_device('') # 0 for gpu, '' for cpu
# initialize deepsort
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort('osnet_x0_25',
                    device,
                    max_dist=cfg.DEEPSORT.MAX_DIST,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    )
# Get names and colors
count = 0
count_id = []

names = model.module.names if hasattr(model, 'module') else model.names


from vidgear.gears import CamGear
import cv2, time

# def stream(id):
#     # url = "https://www.youtube.com/watch?v=DOrDtgdZzMo"
#     # time.sleep(2)
#     stream = CamGear(source="https://www.youtube.com/watch?v=DnokJ5jVb40", stream_mode = True, logging=True).start()

#     # stream = cv2.VideoCapture("videos/Traffic.mp4")
#     model.conf = 0.7
#     model.iou = 0.5
#     model.classes = [0]
#     # model.classes = [0,64,39]
#     startTime = 0
#     while True:
#         # ret, frame = stream.read()
#         frame = stream.read()
#         frame = cv2.resize(frame, (500, 250))
#         nowTime = time.time()
#         fps = 1 /(nowTime - startTime)
#         startTime = nowTime
        
#         # frame = cv2.resize(frame, (500,250))
#         # if not ret:
#         #     print("Error: failed to capture image")
#         #     break
        
#         results = model(frame, augment=True)
#         # proccess
#         annotator = Annotator(frame, line_width=2, pil=not ascii) 
#         w, h = frame.shape[1], frame.shape[0]
#         color=(0,255,0)
#         start_point = (0, h-50)
#         end_point = (w, h-50)
#         # cv2.line(frame, start_point, end_point, color, thickness=2)
#         thickness = 1
#         org = (50, 50)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         fontScale = 1
#         # cv2.putText(frame, str(count), org, font, 
#             # fontScale, color, thickness, cv2.LINE_AA)
#         det = results.pred[0]
#         if det is not None and len(det):   
#             xywhs = xyxy2xywh(det[:, 0:4])
#             confs = det[:, 4]
#             clss = det[:, 5]
#             outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), frame)
            
#             if len(outputs) > 0:
#                 for j, (output, conf) in enumerate(zip(outputs, confs)):
                    
#                     bboxes = output[0:4]
#                     id = output[4]
                    
#                     cls = output[5]
#                     # count_deep(bboxes, w, h, id)
#                     c = int(cls)  # integer class
#                     label = f'{id} {names[c]} {conf:.2f}'
#                     annotator.box_label(bboxes, label, color=colors(c, True))
#         else:
#             deepsort.increment_ages()
#         cv2.putText(frame, "fps: " + str(int(fps)), (100,50), font, 
#             fontScale, color, thickness, cv2.LINE_AA)
#         im0 = annotator.result()    
#         image_bytes = cv2.imencode('.jpg', im0)[1].tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n')
#         if id == 1:
#             break
        
    # stream.release()
    # cv2.destroyAllWindows()
def count_deep(box, w, h, id):
    global count, count_id
    center_circle = (int(box[0] + (box[2] - box[0])/2)) , (int(box[1] + (box[3] - box[1])/2))
    if (int(box[1] + (box[3] - box[1])/2)) > (h - 50):
        if id not in count_id:
            count += 1
            count_id.append(id)

# # @login_required(login_url='login')
# # def video_feed(request):
# #     # q = request.GET.get('q')
# #     t = Video_Feed(count, count_id).start()
# #     # if q is None:
# #     #     chay = False
# #     # # thread_video = Thread(target=stream())
# #     # # global count, count_id
# #     # # count = 0
# #     # # count_id = []
# #     # # time.sleep(1)
# #     return StreamingHttpResponse(t, content_type='multipart/x-mixed-replace; boundary=frame') 

# api
apiCapcha = ApiCapcha.as_view()
apiUser = ApiUser.as_view()
apiUserAll = ApiUserAll.as_view()
apiCapchaAll = ApiCapchaAll.as_view()


def choose_funsion(request, pk):
    room = Room.objects.get(id = pk)
    
    name = str(room.name)
    if name == "Cap Cha Tik Tok":
        # return render(request, 'base/capChaTron.html')
        return redirect('capChaTron', pk = room.id)
    
    elif name == "Deep sort":
        return render(request, 'base/video_test.html')
    
    elif name == "Web Cam DeTect":
        return render(request, 'base/video_webcam.html')
    
    else:
        return render(request, 'base/home.html')

# def video_feed(request):
#     id = 0
#     if request.method == 'POST':
#         id = 1
#     return StreamingHttpResponse(stream(id), content_type='multipart/x-mixed-replace; boundary=frame') 


def video_test(dem):
    stream = cv2.VideoCapture("videos/Traffic.mp4")
    model.conf = 0.7
    model.iou = 0.5
    model.classes = [2]
    # model.classes = [0,64,39]
    startTime = 0
    while True:
        ret, frame = stream.read()
        frame = cv2.resize(frame, (500, 250))
        nowTime = time.time()
        fps = 1 /(nowTime - startTime)
        startTime = nowTime
        results = model(frame, augment=True)
        annotator = Annotator(frame, line_width=2, pil=not ascii) 
        w, h = frame.shape[1], frame.shape[0]
        color=(0,255,0)
        start_point = (0, h-50)
        end_point = (w, h-50)
        cv2.line(frame, start_point, end_point, color, thickness=1)
        thickness = 1
        org = (50, 50)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        cv2.putText(frame, str(count), org, font, 
            fontScale, color, thickness, cv2.LINE_AA)
        det = results.pred[0]
        
        
        if det is not None and len(det):   
            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]
            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), frame)
            
            if len(outputs) > 0:
                for j, (output, conf) in enumerate(zip(outputs, confs)):
                    bboxes = output[0:4]
                    id = output[4]
                    cls = output[5]
                    count_deep(bboxes, w, h, id)
                    c = int(cls)  # integer class
                    label = f'{id} {names[c]} {conf:.2f}'
                    annotator.box_label(bboxes, label, color=colors(c, True))
        else:
            deepsort.increment_ages()
        cv2.putText(frame, "fps: " + str(int(fps)), (100,50), font, 
            fontScale, color, thickness, cv2.LINE_AA)
        im0 = annotator.result()    
        image_bytes = cv2.imencode('.jpg', im0)[1].tobytes()
        if dem == 1:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n')

@login_required(login_url='login')
def video_feed_video(request):
    global count, count_id
    id = 0
    if request.method == 'POST':
        id = 1
        count = 0
        count_id = []
    return StreamingHttpResponse(video_test(id), content_type='multipart/x-mixed-replace; boundary=frame') 

@login_required(login_url='login')
def capChaTron(request, pk):
    room = Room.objects.get(id = pk)
    code_pre = None
    img_capCha = None
    # path = 'static/images/img/'
    dem = 0
    if request.method == 'POST':
        name_img = get_random_string(15)
        CapChaTikTokTron(request.FILES.get('image_small').read(),
                            request.FILES.get('image_big').read(),
                            room,
                            request.user,
                            name_img).start()
        while True:
            try:
                time.sleep(0.2)
                img_capCha = CapChaTikTok.objects.get(name = name_img)
                code_pre = img_capCha.codeImg_pre
                dem = 1
            except:
                pass
            if dem == 1:
                break
        print("predict capchatitok tron <main>: ",type(code_pre))
    context = {'room':room, 'img_byte':code_pre}
    return render(request,'base/capChaTron.html',context)
    
@login_required(login_url='login')
def video_feed(request):
    id = 0
    if request.method == 'POST':
        id = 1
    return StreamingHttpResponse(read_from_webcam(id), content_type="multipart/x-mixed-replace; boundary=frame" )
    
    
def read_from_webcam(dem):
    webcam = Webcam()
    model.conf = 0.7
    model.iou = 0.5
    # model.classes = [0]
    model.classes = [0,64,39]
    startTime = 0
    while True:
        # Đọc ảnh từ class Webcam
        image = next(webcam.get_frame())
        
        
        frame = cv2.resize(image, (500, 250))
        nowTime = time.time()
        fps = 1 /(nowTime - startTime)
        startTime = nowTime
        results = model(frame, augment=True)
        annotator = Annotator(frame, line_width=2, pil=not ascii)
        w, h = frame.shape[1], frame.shape[0]
        color=(0,255,0)
        # start_point = (0, h-50)
        # end_point = (w, h-50)
        # cv2.line(frame, start_point, end_point, color, thickness=2)
        thickness = 1
        org = (50, 50)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        # cv2.putText(frame, str(count), org, font, 
        #     fontScale, color, thickness, cv2.LINE_AA)
        det = results.pred[0]
        if det is not None and len(det):   
            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]
            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), frame)
            
            if len(outputs) > 0:
                for j, (output, conf) in enumerate(zip(outputs, confs)):
                    bboxes = output[0:4]
                    id = output[4]
                    cls = output[5]
                    # count_deep(bboxes, w, h, id)
                    c = int(cls)  # integer class
                    label = f'{id} {names[c]} {conf:.2f}'
                    annotator.box_label(bboxes, label, color=colors(c, True))
        else:
            deepsort.increment_ages()
        cv2.putText(frame, "fps: " + str(int(fps)), (100,50), font, 
            fontScale, color, thickness, cv2.LINE_AA)
        im0 = annotator.result()    
        image_bytes = cv2.imencode('.jpg', im0)[1].tobytes()
        if dem == 1:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n')
        