from rest_framework.decorators import api_view, permission_classes
import base64
import io
from PIL import Image
from tensorflow.keras.utils import array_to_img
from rest_framework.response import Response
from rest_framework import status
from .serializer import *
from .funsion_base import *
# from snippets.models import Snippet
# from snippets.serializers import SnippetSerializer
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated


# GET - xem danh sach
# POST - them vao danh sach
# PUT - cap nhat 
# DELETE - xoa
@permission_classes([IsAuthenticated])
class ApiCapchaAll(APIView):
    def get(self, request, format=None):
        obj = CapChaTikTok.objects.all()[0:5]
        myData = ApiCapchaTikTok(obj, many=True)
        return Response(myData.data)
    def post(self, request, format=None):
        # obj_user = User.objects.get(id = 1)
        obj_room = Room.objects.get(id = 2)
        print("===>user post: ", request.user)
        
        code_small = request.data['codeImg_small']
        code_big = request.data['codeImg_big']
        
        nho = base64.b64decode(code_small)
        img_small = Image.open(io.BytesIO(nho))

        lon = base64.b64decode(code_big)
        img_big = Image.open(io.BytesIO(lon))

        small = img_small.convert('RGB')
        big = img_big.convert('RGB')

        small = small.resize((211,211))
        big = big.resize((347,347))

        small_img = np.array(small, dtype= np.uint8)
        big_img = np.array(big, dtype= np.uint8)

        small_img=array_to_img(small_img)
        big_img=array_to_img(big_img)
        
        img_byte = io.BytesIO()
        img = capChaTronTikTok(small_img, big_img)
        img.save(img_byte, format='PNG')
        img_byte = img_byte.getvalue()
        
        code = base64.b64encode(img_byte).decode('utf-8')
        _data = CapChaTikTok.objects.create(
            user = request.user,
            room = obj_room,
            name = get_random_string(15),
            codeImg_small = request.data['codeImg_small'],
            codeImg_big = request.data['codeImg_big'],
            codeImg_pre = code,
        )

        _data.save()
        print(f"===>User {request.user} post thành công!!! ", )
        myData = ApiCapchaTikTok(_data, many=False)
        
        
        # obj = ApiCapchaTikTok(data=request.data)
        # if obj.is_valid():
        #     obj.save()
        #     myData = ApiCapchaTikTok(obj, many=True)
        #     return Response(myData.data, status=status.HTTP_201_CREATED)
        return Response(myData.data, status=status.HTTP_201_CREATED)
        # return Response(myData.errors, status=status.HTTP_400_BAD_REQUEST)



class ApiCapcha(APIView):
    def get_object(self, pk):
        try:
            return CapChaTikTok.objects.get(id=pk)
        except CapChaTikTok.DoesNotExist:
            raise Http404
    def get(self, request, pk, format=None):
        obj = self.get_object(pk)
        myData = ApiCapchaTikTok(obj, many= False)
        return Response(myData.data)
    def put(self, request, pk, format=None):
        myData = {"Lỗi": "Bạn cần quyền admin để thực hiện thao tác này"}
        print("User put capchaTron ", request.user, "với id ",request.user.id)
        if request.user.id == 1:
            try:
                obj = self.get_object(pk)
                obj.name = request.data['name']
                myData = ApiCapchaTikTok(obj, many = False)
                # if myData.is_valid():
                #     myData.save()
                return Response(myData.data)
            except:
                print(f"User {request.user} không có quyền thực hiện thao tác put apicapcha!!")
                return Response(myData)
                # return Response(myData.errors, status=status.HTTP_400_BAD_REQUEST)
    def delete(self, request, pk, format=None):
        myData = {"Lỗi": "Bạn cần quyền admin để thực hiện thao tác này"}
        print("User delete capchaTron ", request.user, "với id ",request.user.id)
        if request.user.id == 1:
            obj = self.get_object(pk)
            obj.delete()
            print(f"User {request.user} thực hiện delete thành công apicapcha!!")
            return Response(status=status.HTTP_204_NO_CONTENT)
        else:
            print(f"User {request.user} không có quyền thực hiện thao tác delete apicapcha!!")
            return Response(myData)


@permission_classes([IsAuthenticated])
class ApiUser(APIView):
    def get_object(self, pk):
        try:
            return User.objects.get(id=pk)
        except User.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        obj = self.get_object(pk)
        myData = ApiUserSer(obj, many = False)
        return Response(myData.data)
    def put(self, request, pk, format=None):
        myData = {"Lỗi": "Bạn cần quyền admin để thực hiện thao tác này"}
        print("User put apiUser ", request.user, "với id ",request.user.id)
        if request.user.id == 1:
            try:
                obj = self.get_object(pk)
                obj.name = request.data['name']
                obj.username = request.data['username']
                obj.about = request.data['about']
                myData = ApiUserSer(obj, many = False)
                return Response(myData.data)
            except:
                print("Lỗi put user")
                obj = self.get_object(pk)
                myData = ApiUserSer(obj, many = False)
                return Response(myData.errors, status=status.HTTP_400_BAD_REQUEST)
        else:
            print(f"User {request.user} không có quyền thực hiện thao tác put user!!")
            return Response(myData)
    def delete(self, request, pk, format=None):
        print("User delete apiUser ", request.user, "với id ",request.user.id)
        myData = {"Lỗi": "Bạn cần quyền admin để thực hiện thao tác này"}
        obj = self.get_object(pk)
        if request.user.id == 1:
            obj.delete()
            print(f"User {request.user} đã được xóa!!")
            return Response(status=status.HTTP_204_NO_CONTENT)
        else:
            print(f"User {request.user} không có quyền thực hiện thao tác delete user!!")
            return Response(myData)

    
    
@permission_classes([IsAuthenticated])
class ApiUserAll(APIView):
    def get(self, request, format=None):
        obj = User.objects.all()
        myData = ApiUserSer(obj, many=True)
        return Response(myData.data)
    
    def post(self, request, format=None):
    
        myData = {"Lỗi": "Bạn cần quyền admin để thực hiện thao tác này"}
        print("User post apiUser ", request.user, "với id ",request.user.id)
        if request.user.id == 1:
            print("====> User thao tác:", request.user)
            try:
                obj = User.objects.create(
                        name = request.data['name'],
                        email = request.data['email'],
                        username = request.data['username'],
                )
                obj.set_password(request.data['password'])
                obj.save()
                myData = ApiUserSer(obj, many = False) 
                print("====>Đã tạo User thành công!")
                return Response(myData.data, status=status.HTTP_201_CREATED)
            except:
                # obj = User.objects.get(id = 1)
                print("====>Tạo User thất bại, User post:", request.user)
                # myData = ApiUserSer(obj, many = False) 
                return Response(myData)
        else:
            print(f"User {request.user} không có quyền thực hiện thao tác post user!!")
            return Response(myData)
        # return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# @api_view(['GET', 'DELETE'])
# def getCapcha(request, pk):
#     obj = CapChaTikTok.objects.all()
#     query = request.GET.get('query')
#     if query == None:
#         if request.method == 'GET':
#             myData = ApiCapchaTikTok(obj, many= True)
#             return Response(data=myData.data, status=status.HTTP_200_OK)
#         else:
#             if request.method == 'GET':
#                 obj = CapChaTikTok.objects.get(id = pk)
#                 myData = ApiCapchaTikTok(obj, many= True)
#                 return Response(data=myData.data, status=status.HTTP_200_OK)
#     if request.method == 'DELETE':
#         obj = CapChaTikTok.objects.get(id = pk)
#         obj.delete()
#         return redirect('getCapcha')


# @api_view(['GET', 'POST'])
# @permission_classes([IsAuthenticated])
# def getUser(request):
#     if request.method == 'GET':
#         obj = User.objects.all()[0:5]
#         myData = ApiUserSer(obj, many= True)
#         return Response(data=myData.data, status=status.HTTP_200_OK)
    
#     if request.method == 'POST':
#         obj = User.objects.create(
#                 name = request.data['name'],
#                 email = request.data['email'],
#                 username = request.data['username'],
#         )
#         obj.set_password(request.data['password'])
#         obj.save()
#         myData = ApiUserSer(obj, many = False)
#         return Response(myData.data)
    
    