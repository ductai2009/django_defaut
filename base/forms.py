
from .models import Room, User, image_64, CapChaTikTok
from django.forms import ModelForm
from django.contrib.auth.forms import UserCreationForm


class CreateUser(UserCreationForm):
    class Meta:
        model = User
        fields = ['name','username', 'email','password1','password2']
      
class UpLoadImg(ModelForm):
    class Meta:
        model = image_64
        fields = ['name','image']
    
class RoomForm(ModelForm):
    class Meta:
        model = Room
        fields = '__all__'
        exclude = ['host', 'participants']

class UserForm(ModelForm):
    class Meta:
        model = User
        fields = ['username','name', 'email', 'avatar','about']
        

  
