from django import forms
from .models import Users, UsersIdentificationphotos, Events, Headbands, User_Photo


class AddUserPhotoForm(forms.ModelForm):

    class Meta:
        model = UsersIdentificationphotos
        fields = ('identification_photo_1', 'identification_photo_2')
        widgets = {
            'identification_photo_1': forms.FileInput(attrs={'placeholder': 'Добавьте файл'}),
            'identification_photo_2': forms.FileInput(attrs={'placeholder': 'Добавьте файл'}),
        }

class AddUserProfilePhotoForm(forms.ModelForm):
    class Meta:
        model = User_Photo
        fields = ('photo',)
        widgets = {
            'photo': forms.FileInput(attrs={'placeholder': 'Добавьте фото профиля'}),
        }

class ChangeUserPhotoForm(forms.ModelForm):
    class Meta:
        model = UsersIdentificationphotos
        fields = ('identification_photo_1', 'identification_photo_2')
        widgets = {
            'identification_photo_1': forms.FileInput(attrs={'placeholder': 'Добавьте файл'}),
            'identification_photo_2': forms.FileInput(attrs={'placeholder': 'Добавьте файл'}),
        }


class AddPreviewPhotoForm(forms.ModelForm):

    class Meta:
        model = Headbands
        fields = ('headband', )
        widgets = {
            'headband': forms.ClearableFileInput(attrs={'placeholder': 'Добавьте файл'}),
        }


class ChangePreviewPhotoForm(forms.ModelForm):

    class Meta:
        model = Headbands
        fields = ('headband', )
        widgets = {
            'headband': forms.ClearableFileInput(attrs={'placeholder': 'Добавьте файл'}),
        }


class ConfirmEventPasswordForm(forms.Form):
    password = forms.CharField(label='Пароль', widget=forms.TextInput(attrs={'placeholder': 'Введите пароль'}))


class AddEvent(forms.ModelForm):
    class Meta:
        model = Events
        fields = ['eventname', 'description', 'eventdate', 'location', 'file_archive', 'password', 'unique_link', 'private_mode']
        widgets = {
            'eventname': forms.TextInput(),
            'description': forms.Textarea(),
            'eventdate': forms.DateInput(),
            'location': forms.TextInput(),
            'file_archive': forms.FileInput(),
            'password': forms.TextInput(),
            'unique_link': forms.TextInput(),
            'private_mode': forms.CheckboxInput(),
        }


