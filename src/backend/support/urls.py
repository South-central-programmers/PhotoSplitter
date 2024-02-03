from django.contrib.auth.views import PasswordResetView, PasswordResetDoneView, PasswordResetConfirmView, \
    PasswordResetCompleteView
from django.urls import path
from .views import *

urlpatterns = [
    path('', main_page, name='main'),
    path('about/', about, name='about'),
    path('contacts/', contacts, name='contacts'),
    path('find/',  search_events_paginator, name='find'),
    path('profile/', go_profile, name='profile'),
    path('download_all_zip/<int:event_id>', download_all_zip, name='download_all_zip'),
    path('confirm_password/<int:event_id>', confirm_password, name='confirm_password'),
    path('add_event/', add_event, name='add_event'),
    path('registration/', register, name='reg'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('add_photo/', add_profile_photos, name='add_photo'),
    path('change_photo/', change_profile_photos, name='change_photo'),
    path('add_preview/<int:event_id>', add_preview_photos, name='add_preview'),
    path('change_preview/<int:event_id>', change_preview_photos, name='change_preview'),
    path('my_events/<int:user_id>', check_events, name='my_events'),
    path('detail_events/<int:event_id>', view_detail_events, name='detail_events'),
    path('password-reset/', PasswordResetView.as_view(template_name='support/password_reset_form.html'), name='password_reset'),
    path('password-reset/done/', PasswordResetDoneView.as_view(template_name='support/password_reset_done.html'), name='password_reset_done'),
    path('password-reset/<uidb64>/<token>/', PasswordResetConfirmView.as_view(template_name='support/password_reset_confirm.html'), name='password_reset_confirm'),
    path('password-reset/complete/', PasswordResetCompleteView.as_view(template_name='support/password_reset_complete.html'), name='password_reset_complete'),
]

