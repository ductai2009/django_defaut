
from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    name = models.CharField(max_length=100, null= True)
    email = models.EmailField(null=True, unique = True)
    about = models.TextField(max_length=200, null = True)
    avatar = models.ImageField(null = True, default = 'avatar.svg')
    
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS =  ['username']
    

# Create your models here.
class Topic(models.Model):
    name = models.CharField(max_length = 200)
    def __str__(self):
        return self.name
    
class Room(models.Model):
    host = models.ForeignKey(User,on_delete=models.SET_NULL, null=True)
    topic = models.ForeignKey(Topic, on_delete=models.SET_NULL, null=True)
    name = models.CharField(max_length=100)
    participants = models.ManyToManyField(User, blank = True, related_name="participan")
    description = models.TextField(null=True, blank=True)
    updated = models.DateTimeField(auto_now=True)
    created = models.DateTimeField(auto_now_add= True)
    
    class Meta:
        # sắp xếp theo thứ tự mới đến cũ của các bài đăng 
        ordering = ['-updated', '-created']
    def __str__(self):
        return self.name
    
class image_64(models.Model):
    user = models.ForeignKey(User,on_delete=models.CASCADE)
    room = models.ForeignKey(Room,on_delete=models.SET_NULL, null = True)
    name = models.CharField(max_length=200, default= 'null')
    codeImg = models.TextField(max_length=300000, default= 'null')
    image = models.ImageField(null = True, default = 'avatar.svg')
    
    def __str__(self):
        return self.name
    
    
class Mesenge(models.Model):
    # one room have many mesange
    user = models.ForeignKey(User,on_delete=models.CASCADE)
    room = models.ForeignKey(Room,on_delete=models.CASCADE)
    body = models.TextField(max_length=100)
    updated = models.DateTimeField(auto_now=True)
    created = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return self.body[0:50]
    class Meta:
        # sắp xếp theo thứ tự mới đến cũ của các bài đăng 
        ordering = ['-updated', '-created']
      
           
class CapChaTikTok(models.Model):
    # one room have many mesange
    user = models.ForeignKey(User,on_delete=models.CASCADE)
    room = models.ForeignKey(Room,on_delete=models.CASCADE)
    name = models.CharField(max_length=200, default= 'null')
    codeImg_small = models.TextField(max_length=300000, default= 'null')
    codeImg_big = models.TextField(max_length=300000, default= 'null')
    updated = models.DateTimeField(auto_now=True)
    created = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return self.name
    class Meta:
        # sắp xếp theo thứ tự mới đến cũ của các bài đăng 
        ordering = ['-updated', '-created']
    