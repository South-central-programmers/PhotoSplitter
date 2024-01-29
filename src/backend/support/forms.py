from django import forms
from .models import Users, UsersIdentificationphotos, Events
from django.contrib.auth.models import User

class EventsSearchForm(forms.Form):
    query = forms.TextInput()
    def clean_query(self):
        query = self.cleaned_data.get('query')
        if not query:
            raise forms.ValidationError('Введите поисковый запрос')
        return query


class UserRegistrationForm(forms.ModelForm):
    password = forms.CharField(label='Password', widget=forms.PasswordInput(attrs={'placeholder': 'Введите ваш пароль'}))
    password2 = forms.CharField(label='Repeat password', widget=forms.PasswordInput(attrs={'placeholder': 'Введите ваш пароль повторно'}))
    class Meta:
        model = User
        fields = ('username', 'first_name', 'email', 'last_name')
        widgets = {
            'email': forms.EmailInput(attrs={'placeholder': 'Введите вашу почту'}),
            'first_name' : forms.TextInput(attrs={'placeholder': 'Введите ваше имя'}),
            'last_name': forms.TextInput(attrs={'placeholder': 'Введите вашу фамилию'}),
            'username': forms.TextInput(attrs={'placeholder': 'Введите ваш никнейм'}),
        }

    def clean_password2(self):
        cd = self.cleaned_data
        if cd['password'] != cd['password2']:
            raise forms.ValidationError('Passwords don\'t match.')
        return cd['password2']


class AddUserPhotoForm(forms.ModelForm):

    class Meta:
        model = UsersIdentificationphotos
        fields = ('identification_photo_archive', )
        widgets = {
            'identification_photo_archive': forms.FileInput(attrs={'placeholder': 'Добавьте файл'}),
        }


class AddEvent(forms.ModelForm):
    class Meta:
        model = Events
        fields = ['eventname', 'description', 'eventdate', 'location', 'likes', 'file_archive', 'password', 'unique_link', 'private_mode']
        widgets = {
            'eventname': forms.TextInput(),
            'description': forms.Textarea(),
            'eventdate': forms.DateInput(),
            'location': forms.TextInput(),
            'likes': forms.NumberInput(),
            'file_archive': forms.FileInput(),
            'password': forms.TextInput(),
            'unique_link': forms.TextInput(),
            'private_mode': forms.CheckboxInput(),
        }

