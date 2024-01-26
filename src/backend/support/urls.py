from django.urls import path
from .views import *

urlpatterns = [
    path('', index, name='main'),
    path('profile/', go_profile, name='profile'),
    path('add_event/', add_event, name='add_event'),
    path('registration/', register, name='reg'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('add_photo/', add_profile_photos, name='add_photo'),
    path('my_events/<int:user_id>', check_events, name='my_events'),
]

