# from django.contrib import admin
# from support.models import Users
# from django.contrib.auth.models import User
# from django.contrib.auth.admin import UserAdmin
#
# from .models import Users, Events, EventParticipation
#
#
# class UsersInline(admin.StackedInline):
#     model = Users
#     can_delete = False
#     verbose_name = 'Users'
#
#
# class CustomizedUsers(UserAdmin):
#     inlines = (UsersInline, )
#
#
# admin.site.unregister(User)
# admin.site.register(User, CustomizedUsers)
# admin.site.register(Events)
# admin.site.register(EventParticipation)

from django.contrib import admin
from support.models import Users
from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin
from django.contrib.admin import ModelAdmin
from .models import *

from .models import Users, Events, EventParticipation


class EventsAdmin(ModelAdmin):
    list_display = (['user', 'eventname', 'description', 'eventdate', 'location', 'likes', 'file_archive', 'private_mode'])
    verbose_name = 'Events'


class UsersSavedArchivesAdmin(ModelAdmin):
    list_display = (['user', 'saved_file_archive'])
    verbose_name = 'UsersArchive'
    can_delete = False


class UsersInline(admin.StackedInline):
    model = Users
    can_delete = False
    list_display = (['user', 'email', 'first_name', 'last_name',])
    verbose_name = 'Users'


class CustomizedUsers(UserAdmin):
    inlines = (UsersInline, )


admin.site.unregister(User)
admin.site.register(UsersSavedArchives, UsersSavedArchivesAdmin)
admin.site.register(User, CustomizedUsers)
admin.site.register(Events, EventsAdmin)
admin.site.register(EventParticipation)
admin.site.site_titlle = 'Admin-pannel PhotoSplitter'
admin.site.site_header = 'Admin-pannel PhotoSplitter'
