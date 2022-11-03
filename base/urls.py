from django.urls import path, include
from . import views

from rest_framework.decorators import api_view
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)



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
    path('choose_funsion/<str:pk>', views.choose_funsion, name = 'choose_funsion'),
    path('video_feed', views.video_feed, name = 'video_feed'),
    path('video_feed_test', views.video_feed_video, name = 'video_feed_test'),
    path('capChaTron/<str:pk>', views.capChaTron, name = 'capChaTron'),
    path('restApi_CapChaTron/<str:pk>', views.apiCapcha, name = 'apiCapchaTron'),
    path('restApi_CapChaTronAll/', views.apiCapchaAll, name = 'apiCapchaTronAll'),
    path('restApi_User/<str:pk>', views.apiUser, name = 'apiUser'),
    path('restApi_UserAll/', views.apiUserAll, name = 'apiUserAll'),
    # path('restApi_UserAll_e/', views.apiCapchaAll, name = 'apiUserAll'),
    path('token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    # path('restApi_User/', views.getUser),
    # path('restApi_CapchaTron/<str:pk>/', views.getCapcha, name ='getCapcha'),
]
