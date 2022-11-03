from rest_framework.serializers import ModelSerializer, SerializerMethodField
from .models import *
from rest_framework import serializers


class ApiUserSer(ModelSerializer):
    class Meta:
        write_only_fields = ('password',)
        read_only_fields = ('id',)
        model = User
        # fields = '__all__'
        fields = ['id', 'name', 'email', 'username', 'about']

class ApiCapchaTikTok(ModelSerializer):
    # count_img = SerializerMethodField(read_only = True)
    user = ApiUserSer()
    class Meta:
        read_only_fields = ('id',)
        model = CapChaTikTok
        fields = ['user', 'room', 'name', 'codeImg_small', 'codeImg_big', 'codeImg_pre']
    # def get_count_img(self, obj):
    #     count = obj.CapChaTikTok_set.count()
    #     return count
    
    
        
