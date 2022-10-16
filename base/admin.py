from django.contrib import admin

# Register your models here.
from .models import Room, Topic, Mesenge, User, image_64, CapChaTikTok

admin.site.register(Room)
admin.site.register(Topic)
admin.site.register(Mesenge)
admin.site.register(User)
admin.site.register(image_64)
admin.site.register(CapChaTikTok)