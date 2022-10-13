from django.urls import path
from . import views


urlpatterns = [
    
    path('home/', views.home, name = 'home'),
    path('room/<str:pk>/', views.room, name = 'room'),
    path('create_room/', views.create_room, name = 'create_room'),
    path('update_room/<str:pk>/', views.updateRoom, name = 'update_room'),
    path('detele_room/<str:pk>/', views.deteleRoom, name = 'detele_room'),
    path('', views.logIn, name = 'login'),
    path('logout/', views.logOut, name = 'logout'),
    path('signup/', views.signUp, name = 'signup'),
    path('delete-mes/<str:pk>', views.deteleMessange, name = 'delete-mes'),
    path('profile/<str:pk>', views.profile, name = 'profile'),
    path('updateUser/<str:pk>', views.updateUser, name = 'updateuser'),
    path('topicPage/', views.topicPage, name = 'topicPage'),
    path('activityUser/', views.activityUser, name = 'activityUser'),
    path('upimage/<str:pk>', views.upImage, name = 'UpLoadImg'),
    path('upimg/<str:pk>', views.upimg, name = 'upimg'),
    path('predict/<str:pk>', views.predict, name = 'predict'),
    path('deep_sort/<str:pk>', views.deep_sort, name = 'deep_sort'),
    path('video_feed', views.video_feed, name = 'video_feed'),
    path('video_feed_test', views.video_feed_video, name = 'video_feed_test'),
]
