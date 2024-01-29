from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.shortcuts import render, redirect
from django.views.generic import ListView

from .forms import *
from .models import Users, UsersIdentificationphotos, Events
from django.contrib.auth import authenticate, login, logout


menu = [{'title': 'Домой', 'url_name': 'main'}, {'title': 'Регистрация', 'url_name': 'reg'}, {'title': 'Войти', 'url_name': 'login'}]

def main_page(request):
    #print(request.user.id)
    meetings = Events.objects.all()#.filter(id__lte=3).order_by('id')     #get_queryset().order_by('id') #filter(id__lte=5).order_by('id')
    paginator = Paginator(meetings, 2)
    page_number =   request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    page = request.GET.get('page')
    try:
        meetings = paginator.page(page)
    except PageNotAnInteger:
        meetings = paginator.page(1)
    except EmptyPage:
        meetings = paginator.page(paginator.num_pages)
    return render(request, 'support/main_page.html', {'menu': menu, 'page_obj' : page_obj,    'meetings': meetings})

def search_events_paginator(request):
    context = {}
    meetings = Events.objects.all()
    if request.method == 'GET':
        query = request.GET.get('meetings')
        queryset = meetings.filter(eventname__icontains=query)
        page = request.GET.get('page')
        paginator = Paginator(queryset, 2)
        try:
            meetings = paginator.page(page)
        except PageNotAnInteger:
            meetings = paginator.page(1)
        except EmptyPage:
            meetings = paginator.page(paginator.num_pages)
        total = queryset.count()
        context.update({'meetings': meetings,
                        'total': total,
                        'query': query,
                        'menu' : menu,
                        })
        return render(request, 'support/main_page.html', context)
def event_info(request, event_id):
    event = Events.objects.get(id=event_id)
    return render(request, 'support/event_info.html', {'event': event, 'menu' : menu})
def return_to_main(request):
    return render(request, template_name='support/main.html', context={'menu': menu})


def register(request):
    if request.user.is_authenticated:
        return redirect('main')
    else:
        if request.method == 'POST':
            user_form = UserRegistrationForm(request.POST)
            if user_form.is_valid():
                new_user = user_form.save(commit=False)
                new_user.set_password(user_form.cleaned_data['password'])
                new_user.save()
                return redirect('login')
        else:
            user_form = UserRegistrationForm()
        return render(request, 'support/reg.html', {'user_form': user_form, 'menu' : menu})


def login_view(request):
    if request.user.is_authenticated:
        # return render(request, 'support/profile.html', {'user_id': request.user.id})
        return redirect('main')
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            # return render(request, 'support/profile.html', {'user_id': request.user.id})
            return redirect('main')
        else:
            error_message = 'Пользователь не найден'
            return render(request, 'support/login.html', {'error_message': error_message, 'menu' : menu})
    else:
        return render(request, template_name='support/login.html', context={'menu' : menu})


def add_profile_photos(request):
    if request.method == 'POST':
        form = AddUserPhotoForm(request.POST, request.FILES)
        if form.is_valid():
            UsersIdentificationphotos.objects.create(identification_photo_archive=form.cleaned_data['identification_photo_archive'], user_id=request.user.id)
        else:
            print('Error')
    else:
        form = AddUserPhotoForm()
    # return render(request, 'support/add_photo.html', {'user_id': request.user.id})
    return render(request, 'support/add_photo.html', {'form': form, 'menu' : menu})


def logout_view(request):
    if not request.user.is_authenticated:
        return redirect('login')
        # return render(request, 'support/login.html', {'error_message': 'Сначала зарегистрируйтесь!'})

    logout(request)
    return redirect('main')


def add_event(request):
    if request.method == 'POST':
        form = AddEvent(request.POST, request.FILES)
        if form.is_valid():
            Events.objects.create(eventname=form.cleaned_data['eventname'], description=form.cleaned_data['description'],
                                  eventdate=form.cleaned_data['eventdate'], location=form.cleaned_data['location'],
                                  likes=form.cleaned_data['likes'], file_archive=form.cleaned_data['file_archive'],
                                  password=form.cleaned_data['password'], unique_link=form.cleaned_data['unique_link'],
                                  private_mode=form.cleaned_data['private_mode'], user_id=request.user.id)
        else:
            print('Error')
    else:
        form = AddEvent()
    return render(request, 'support/add_event.html', {'form': form, 'menu' : menu})


def go_profile(request):
    return render(request, 'support/profile.html', {'user_id': request.user.id, 'menu' : menu})


def go_to_add_photo(request):
    return render(request, 'support/add_photo.html', {'user_id': request.user.id, 'menu' : menu})


def add_gallery(request):
    return render(request, template_name='support/gallery.html', context = {'menu' : menu})


def check_events(request, user_id):
    events = Events.objects.filter(user_id=request.user.id)
    return render(request, 'support/my_events.html', {'events': events, 'menu' : menu})


