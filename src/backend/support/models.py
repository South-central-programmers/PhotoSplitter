import os
from django.db import models
from django.contrib.auth.models import User


class Users(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, default=None)
    email = models.EmailField(max_length=50, unique=True, default=None)
    first_name = models.CharField(max_length=50, default=None)
    last_name = models.CharField(max_length=50, default=None)

    def __str__(self):
        return self.user.username


def users_identification_photos_path(instance, filename):
    return os.path.join('upload_user_files', str(instance.user.id), filename)
    # return 'upload_profile_files/{0}/{1}'.format(instance.user.id, filename)


def events_path(instance, filename):
    return os.path.join('upload_event_files', str(instance.user.id), str(instance.user.id))


class UsersIdentificationphotos(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, default=None)
    identification_photo_archive = models.FileField(upload_to=users_identification_photos_path)
    # id_of_user = models.IntegerField(default=None)


class UsersSavedArchives(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, default=None)
    saved_file_archive = models.FileField(upload_to='upload_user_files')
    # id_of_user = models.IntegerField(default=None)


class Events(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, default=None)
    eventname = models.CharField(max_length=50)
    description = models.TextField(max_length=150)
    eventdate = models.CharField(max_length=30)
    location = models.CharField(max_length=150)
    likes = models.IntegerField(default=0)
    file_archive = models.FileField(upload_to=events_path)
    password = models.CharField(max_length=30, blank=True)
    unique_link = models.URLField(blank=True)
    private_mode = models.BooleanField(default=False)


class EventParticipation(models.Model):
    event_id = models.IntegerField()
    user_id = models.IntegerField()





