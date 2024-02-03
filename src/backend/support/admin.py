from django.contrib import admin
from support.models import Users
from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin

from .models import Users, Events, EventParticipation


class UsersInline(admin.StackedInline):
    model = Users
    can_delete = False
    verbose_name = "Users"


class CustomizedUsers(UserAdmin):
    inlines = (UsersInline,)


admin.site.unregister(User)
admin.site.register(User, CustomizedUsers)
admin.site.register(Events)
admin.site.register(EventParticipation)
