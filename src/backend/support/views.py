import os
import zipfile
import rarfile
import tarfile
import py7zr
from pathlib import Path

from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage

from .forms import (
    UserRegistrationForm,
    AddUserPhotoForm,
    AddEvent,
    AddPreviewPhotoForm,
    ChangeUserPhotoForm,
    ChangePreviewPhotoForm,
    ConfirmEventPasswordForm,
)
from .models import (
    UsersIdentificationphotos,
    Events,
    PathToEventsFiles,
    Headbands,
    PathToArchiveRelease,
    User,
)

BASE_DIR = Path(__file__).resolve().parent.parent

menu = [
    {"title": "Домой", "url_name": "main"},
    {"title": "Выход", "url_name": "logout"},
    {"title": "Добавить событие", "url_name": "add_event"},
    {"title": "Профиль", "url_name": "profile"},
]


class EventObjects:
    def __init__(self, info_event, event_photos):
        self.info_event = info_event
        self.event_photos = event_photos


def about(request):
    menu_local = [
        {"title": "Регистрация", "url_name": "reg"},
        {"title": "Войти", "url_name": "login"},
        {"title": "Контакты", "url_name": "contacts"},
    ]
    return render(request, "support/about.html", {"menu": menu_local})


def contacts(request):
    menu_local = [
        {"title": "Регистрация", "url_name": "reg"},
        {"title": "Войти", "url_name": "login"},
    ]
    return render(request, "support/contacts.html", {"menu": menu_local})


def main_page(request):
    if not request.user.is_authenticated:
        return redirect("about")
    events = Events.objects.all().order_by("-likes")
    paginator = Paginator(events, 2)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)
    page = request.GET.get("page")
    try:
        events = paginator.page(page)
    except PageNotAnInteger:
        events = paginator.page(1)
    except EmptyPage:
        events = paginator.page(paginator.num_pages)
    absolute_objects = []
    for elem in events:
        try:
            this_event_preview = (
                f"/media/{Headbands.objects.get(event_id=elem.id).headband}"
            )
        except:
            this_event_preview = "/media/headbands_of_events/default/Хочу пятёрку.jpg"
        one_object = EventObjects(elem, this_event_preview)
        print(one_object.info_event)
        absolute_objects.append(one_object)
    for elem in events:
        print(elem)
    return render(
        request,
        "support/main_page.html",
        {
            "menu": menu,
            "page_obj": page_obj,
            "absolute_objects": absolute_objects,
            "events": events,
            "user_id": request.user.id,
        },
    )


def search_events_paginator(request):
    context = {}
    events = Events.objects.all().order_by("-likes")
    if request.method == "GET":
        query = request.GET.get("meetings")
        queryset = events.filter(eventname__icontains=query)
        page = request.GET.get("page")
        paginator = Paginator(queryset, 2)
        try:
            events = paginator.page(page)
        except PageNotAnInteger:
            events = paginator.page(1)
        except EmptyPage:
            events = paginator.page(paginator.num_pages)
        total = queryset.count()
        absolute_objects = []
        for elem in events:
            try:
                this_event_preview = (
                    f"/media/{Headbands.objects.get(event_id=elem.id).headband}"
                )
            except:
                this_event_preview = (
                    "/media/headbands_of_events/default/Хочу пятёрку.jpg"
                )
            one_object = EventObjects(elem, this_event_preview)
            print(one_object.info_event)
            absolute_objects.append(one_object)
        for elem in events:
            print(elem)
        context.update(
            {
                "events": events,
                "absolute_objects": absolute_objects,
                "total": total,
                "query": query,
                "menu": menu,
            }
        )

        return render(request, "support/main_page.html", context)


def register(request):
    if request.user.is_authenticated:
        return redirect("main")
    else:
        if request.method == "POST":
            user_form = UserRegistrationForm(request.POST)
            if user_form.is_valid():
                try:
                    User.objects.get(user_form.cleaned_data["email"])
                    new_user = user_form.save(commit=False)
                    new_user.set_password(user_form.cleaned_data["password"])
                    new_user.save()
                    return redirect("login")
                except:
                    print("Error")
        else:
            user_form = UserRegistrationForm()
        return render(
            request, "support/reg.html", {"user_form": user_form, "menu": menu}
        )


def login_view(request):
    if request.user.is_authenticated:
        return redirect("main")
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("main")
        else:
            error_message = "Пользователь не найден"
            return render(
                request, "support/login.html", {"error_message": error_message}
            )
    else:
        return render(request, template_name="support/login.html")


def add_profile_photos(request):
    if request.method == "POST":
        form = AddUserPhotoForm(request.POST, request.FILES)
        if form.is_valid():
            UsersIdentificationphotos.objects.create(
                identification_photo_1=form.cleaned_data["identification_photo_1"],
                identification_photo_2=form.cleaned_data["identification_photo_2"],
                user_id=request.user.id,
            )
            return redirect("profile")
        else:
            print("Error")
    else:
        form = AddUserPhotoForm()
    return render(request, "support/add_photo.html", {"form": form})


def change_profile_photos(request):
    if request.method == "POST":
        form = ChangeUserPhotoForm(request.POST, request.FILES)
        if form.is_valid():
            if (form.cleaned_data["identification_photo_1"] is not None) and (
                form.cleaned_data["identification_photo_2"] is None
            ):
                this_obj = UsersIdentificationphotos.objects.get(
                    user_id=request.user.id
                )
                full_file_path = os.path.join(
                    BASE_DIR, f"media/{str(this_obj.identification_photo_1)}"
                )
                if os.path.exists(full_file_path):
                    os.remove(full_file_path)
                this_obj.identification_photo_1 = form.cleaned_data[
                    "identification_photo_1"
                ]
                this_obj.save()
                return redirect("profile")
            elif (form.cleaned_data["identification_photo_1"] is None) and (
                form.cleaned_data["identification_photo_2"] is not None
            ):
                this_obj = UsersIdentificationphotos.objects.get(
                    user_id=request.user.id
                )
                full_file_path = os.path.join(
                    BASE_DIR, f"media/{str(this_obj.identification_photo_2)}"
                )
                if os.path.exists(full_file_path):
                    os.remove(full_file_path)
                this_obj.identification_photo_2 = form.cleaned_data[
                    "identification_photo_2"
                ]
                this_obj.save()
                return redirect("profile")
            elif (form.cleaned_data["identification_photo_1"] is not None) and (
                form.cleaned_data["identification_photo_2"] is not None
            ):
                this_obj = UsersIdentificationphotos.objects.get(
                    user_id=request.user.id
                )
                full_file_path_1 = os.path.join(
                    BASE_DIR, f"media/{str(this_obj.identification_photo_1)}"
                )
                full_file_path_2 = os.path.join(
                    BASE_DIR, f"media/{str(this_obj.identification_photo_2)}"
                )
                if os.path.exists(full_file_path_1):
                    os.remove(full_file_path_1)
                if os.path.exists(full_file_path_2):
                    os.remove(full_file_path_2)
                this_obj.identification_photo_1 = form.cleaned_data[
                    "identification_photo_1"
                ]
                this_obj.identification_photo_2 = form.cleaned_data[
                    "identification_photo_2"
                ]
                this_obj.save()
                return redirect("profile")
        else:
            print("Error")
    else:
        form = ChangeUserPhotoForm()
    return render(request, "support/change_photo.html", {"form": form})


def add_preview_photos(request, event_id):
    if request.method == "POST":
        form = AddPreviewPhotoForm(request.POST, request.FILES)
        if form.is_valid():
            Headbands.objects.create(
                headband=form.cleaned_data["headband"], event_id=event_id
            )
            return redirect("main")
        else:
            print("Error")
    else:
        form = AddPreviewPhotoForm()
    return render(
        request, "support/add_preview.html", {"form": form, "event_id": event_id}
    )


def change_preview_photos(request, event_id):
    if request.method == "POST":
        form = ChangePreviewPhotoForm(request.POST, request.FILES)
        if form.is_valid():
            this_obj = Headbands.objects.get(event_id=event_id)
            full_file_path = os.path.join(BASE_DIR, f"media/{str(this_obj.headband)}")
            if os.path.exists(full_file_path):
                os.remove(full_file_path)
            this_obj.headband = form.cleaned_data["headband"]
            this_obj.save()
            return redirect("main")
        else:
            print("Error")
    else:
        form = ChangePreviewPhotoForm()
    return render(
        request, "support/change_preview.html", {"form": form, "event_id": event_id}
    )


def logout_view(request):
    if not request.user.is_authenticated:
        return redirect("login")

    logout(request)
    return redirect("main")


def get_compression_type(file_path):
    try:
        with zipfile.ZipFile(file_path) as zf:
            return "zip"
    except zipfile.BadZipFile:
        pass

    try:
        with rarfile.RarFile(file_path) as rf:
            return "rar"
    except rarfile.Error:
        pass

    try:
        with py7zr.SevenZipFile(file_path) as rf:
            return "7z"
    except:
        pass

    try:
        with tarfile.open(file_path) as rf:
            return "gz"
    except:
        pass

    return "Unknown"


def dearchive_zip_file(cutted_file_path, id_of_event, full_file_path):
    if get_compression_type(full_file_path) == "zip":
        file_zip = zipfile.ZipFile(full_file_path, "r")
        file_zip.extractall(
            os.path.join(BASE_DIR, f"media/{cutted_file_path}/{id_of_event}")
        )


def dearchive_rar_file(
    cutted_file_path, id_of_event, full_file_path, clear_name_of_archive
):
    file_rar = rarfile.RarFile(full_file_path, "r")
    file_rar.extractall(
        os.path.join(BASE_DIR, f"media/{cutted_file_path}/{id_of_event}")
    )


def dearchive_7z_file(file_path, cutted_file_path, id_of_event, full_file_path):
    file_seven_zip = py7zr.SevenZipFile(full_file_path, "r")
    file_seven_zip.extractall(
        os.path.join(BASE_DIR, f"media/{cutted_file_path}/{id_of_event}")
    )


def dearchive_targz_file(cutted_file_path, id_of_event, full_file_path):
    with tarfile.open(f"{full_file_path}", "r:gz") as tar:
        tar.extractall(
            os.path.join(BASE_DIR, f"media/{cutted_file_path}/{id_of_event}")
        )


def add_event(request):
    if request.method == "POST":
        form = AddEvent(request.POST, request.FILES)
        if form.is_valid() and (
            (
                form.cleaned_data["private_mode"] == 1
                and form.cleaned_data["password"] != ""
            )
            or (
                form.cleaned_data["private_mode"] == 1
                and form.cleaned_data["unique_link"] != ""
            )
            or (
                form.cleaned_data["private_mode"] == 0
                and form.cleaned_data["password"] == ""
                and form.cleaned_data["password"] == ""
            )
        ):
            Events.objects.create(
                eventname=form.cleaned_data["eventname"],
                description=form.cleaned_data["description"],
                eventdate=form.cleaned_data["eventdate"],
                location=form.cleaned_data["location"],
                likes=form.cleaned_data["likes"],
                file_archive=form.cleaned_data["file_archive"],
                password=form.cleaned_data["password"],
                unique_link=form.cleaned_data["unique_link"],
                private_mode=form.cleaned_data["private_mode"],
                user_id=request.user.id,
            )
            queryset_of_last_event = Events.objects.latest("id")
            part_of_set = queryset_of_last_event.file_archive
            full_file_path = os.path.join(BASE_DIR, f"media/{str(part_of_set)}")
            file_path = str(part_of_set)[0 : str(part_of_set).rfind("/")]
            if get_compression_type(full_file_path) == "zip":
                dearchive_zip_file(
                    file_path, str(queryset_of_last_event.id), full_file_path
                )
                PathToEventsFiles.objects.create(
                    path_to_unarchive_file=f"{file_path}/{str(queryset_of_last_event.id)}",
                    id_of_event=str(queryset_of_last_event.id),
                    id_of_user=request.user.id,
                    clear_name_of_archive="None",
                    type_of_archive="zip",
                )
            elif get_compression_type(full_file_path) == "rar":
                clear_name_of_archive = str(part_of_set)[
                    str(part_of_set).rfind("/") + 1 : str(part_of_set).rfind(".")
                ]
                dearchive_rar_file(
                    file_path,
                    str(queryset_of_last_event.id),
                    full_file_path,
                    clear_name_of_archive,
                )
                PathToEventsFiles.objects.create(
                    path_to_unarchive_file=f"{file_path}/{str(queryset_of_last_event.id)}",
                    id_of_event=str(queryset_of_last_event.id),
                    id_of_user=request.user.id,
                    clear_name_of_archive=clear_name_of_archive,
                    type_of_archive="rar",
                )
            elif get_compression_type(full_file_path) == "7z":
                pass
            elif get_compression_type(full_file_path) == "gz":
                dearchive_targz_file(
                    file_path, str(queryset_of_last_event.id), full_file_path
                )
            else:
                print("Bad file")
            print(file_path)
            if os.path.exists(full_file_path):
                os.remove(full_file_path)
            return redirect("main")
        else:
            print("Error")
    else:
        form = AddEvent()
    return render(request, "support/add_event.html", {"form": form})


def go_profile(request):
    try:
        events = Events.objects.filter(user_id=request.user.id)
        absolute_objects = []
        for elem in events:
            all_paths = {}
            this_event_urls = []
            if elem.id not in all_paths:
                this_event = PathToEventsFiles.objects.get(
                    id_of_event=elem.id, id_of_user=request.user.id
                )
                all_paths[str(elem.id)] = this_event.path_to_unarchive_file
                if this_event.type_of_archive == "zip":
                    for key in all_paths:
                        for file_name in os.listdir(
                            os.path.join(BASE_DIR, f"media/{all_paths[key]}")
                        ):
                            this_event_urls.append(
                                f"/media/{all_paths[key]}/{file_name}"
                            )
                if this_event.type_of_archive == "rar":
                    for key in all_paths:
                        for file_name in os.listdir(
                            os.path.join(
                                BASE_DIR,
                                f"media/{all_paths[key]}/{this_event.clear_name_of_archive}",
                            )
                        ):
                            this_event_urls.append(
                                f"/media/{all_paths[key]}/{this_event.clear_name_of_archive}/{file_name}"
                            )
            one_object = EventObjects(elem, this_event_urls)
            absolute_objects.append(one_object)
        result = UsersIdentificationphotos.objects.get(user_id=request.user.id)
        profile_photo_1 = result.identification_photo_1
        profile_photo_2 = result.identification_photo_2
    except:
        print(1)
        profile_photo_info = "Ваш профиль не полный. Добавьте фото"
        return render(
            request,
            "support/profile.html",
            {
                "user_id": request.user.id,
                "profile_photo_info": profile_photo_info,
                "has_full_profile": 0,
            },
        )

    return render(
        request,
        "support/profile.html",
        {
            "user_id": request.user.id,
            "absolute_objects": absolute_objects,
            "profile_photo_1": profile_photo_1,
            "profile_photo_2": profile_photo_2,
            "has_full_profile": 1,
        },
    )


def go_to_add_photo(request):
    return render(
        request,
        "support/add_photo.html",
        {
            "user_id": request.user.id,
        },
    )


def check_events(request, user_id):
    events = Events.objects.filter(user_id=request.user.id)
    absolute_objects = []
    for elem in events:
        all_paths = {}
        this_event_urls = []
        if elem.id not in all_paths:
            this_event = PathToEventsFiles.objects.get(
                id_of_event=elem.id, id_of_user=user_id
            )
            all_paths[str(elem.id)] = this_event.path_to_unarchive_file
            if this_event.type_of_archive == "zip":
                for key in all_paths:
                    for file_name in os.listdir(
                        os.path.join(BASE_DIR, f"media/{all_paths[key]}")
                    ):
                        this_event_urls.append(f"/media/{all_paths[key]}/{file_name}")
            if this_event.type_of_archive == "rar":
                for key in all_paths:
                    for file_name in os.listdir(
                        os.path.join(
                            BASE_DIR,
                            f"media/{all_paths[key]}/{this_event.clear_name_of_archive}",
                        )
                    ):
                        this_event_urls.append(
                            f"/media/{all_paths[key]}/{this_event.clear_name_of_archive}/{file_name}"
                        )
        one_object = EventObjects(elem, this_event_urls)
        absolute_objects.append(one_object)
    return render(
        request, "support/my_events.html", {"absolute_objects": absolute_objects}
    )


def view_detail_events(request, event_id):
    global full_path
    this_event = Events.objects.get(pk=event_id)
    this_event_path_to_catalog = PathToEventsFiles.objects.get(id_of_event=event_id)
    try:
        Headbands.objects.get(event_id=event_id)
        has_headband = 1
    except:
        has_headband = 0
    all_photos = []
    if this_event_path_to_catalog.type_of_archive == "zip":
        full_path = os.path.join(
            BASE_DIR, f"media/{this_event_path_to_catalog.path_to_unarchive_file}"
        )
        for file_name in os.listdir(full_path):
            all_photos.append(
                f"/media/{this_event_path_to_catalog.path_to_unarchive_file}/{file_name}"
            )
    if this_event_path_to_catalog.type_of_archive == "rar":
        full_path = os.path.join(
            BASE_DIR,
            f"media/{this_event_path_to_catalog.path_to_unarchive_file}/{this_event_path_to_catalog.clear_name_of_archive}",
        )
        for file_name in os.listdir(full_path):
            all_photos.append(
                f"/media/{this_event_path_to_catalog.path_to_unarchive_file}/{this_event_path_to_catalog.clear_name_of_archive}/{file_name}"
            )
    print(full_path)
    if request.user.id == this_event.user_id:
        return render(
            request,
            "support/detail_event_this_user.html",
            {
                "this_event": this_event,
                "all_photos": all_photos,
                "has_headband": has_headband,
                "this_event_path_to_catalog": full_path,
            },
        )
    else:
        return render(
            request,
            "support/detail_event.html",
            {"this_event": this_event, "all_photos": all_photos},
        )


def confirm_password(request, event_id):
    if request.method == "POST":
        form = ConfirmEventPasswordForm(request.POST)
        print(form.data)
        if form.is_valid():
            this_event = Events.objects.get(id=event_id)
            real_password = this_event.password
            if real_password == form.data["password"]:
                global full_path
                this_event_path_to_catalog = PathToEventsFiles.objects.get(
                    id_of_event=event_id
                )
                try:
                    Headbands.objects.get(event_id=event_id)
                    has_headband = 1
                except:
                    has_headband = 0
                all_photos = []
                if this_event_path_to_catalog.type_of_archive == "zip":
                    full_path = os.path.join(
                        BASE_DIR,
                        f"media/{this_event_path_to_catalog.path_to_unarchive_file}",
                    )
                    for file_name in os.listdir(full_path):
                        all_photos.append(
                            f"/media/{this_event_path_to_catalog.path_to_unarchive_file}/{file_name}"
                        )
                if this_event_path_to_catalog.type_of_archive == "rar":
                    full_path = os.path.join(
                        BASE_DIR,
                        f"media/{this_event_path_to_catalog.path_to_unarchive_file}/{this_event_path_to_catalog.clear_name_of_archive}",
                    )
                    for file_name in os.listdir(full_path):
                        all_photos.append(
                            f"/media/{this_event_path_to_catalog.path_to_unarchive_file}/{this_event_path_to_catalog.clear_name_of_archive}/{file_name}"
                        )
                print(full_path)
                if request.user.id == this_event.user_id:
                    return render(
                        request,
                        "support/detail_event_this_user.html",
                        {
                            "this_event": this_event,
                            "all_photos": all_photos,
                            "has_headband": has_headband,
                            "this_event_path_to_catalog": full_path,
                        },
                    )
                else:
                    return render(
                        request,
                        "support/detail_event.html",
                        {"this_event": this_event, "all_photos": all_photos},
                    )
            else:
                print("Пароли не совпали")
        else:
            print("Невалидная форма")
    else:
        form = ConfirmEventPasswordForm()

    return render(
        request,
        "support/confirm_event_password.html",
        {"form": form, "this_event": event_id},
    )


def download_all_zip(request, event_id):
    this_event_path_to_catalog = PathToEventsFiles.objects.get(id_of_event=event_id)
    this_event_path_to_catalog_postfix = str(
        this_event_path_to_catalog.path_to_unarchive_file
    )[str(this_event_path_to_catalog.path_to_unarchive_file).find("/") + 1 :]
    full_file_path = os.path.join(
        BASE_DIR, f"media/{str(this_event_path_to_catalog.path_to_unarchive_file)}"
    )
    full_save_file_path = os.path.join(
        BASE_DIR, f"media/archive_release/{str(this_event_path_to_catalog_postfix)}"
    )
    print(full_save_file_path)
    path_to_archive = os.path.join(full_save_file_path, "Your_archive.zip")
    os.makedirs(os.path.dirname(f"{full_save_file_path}/"), exist_ok=True)
    file_zip = zipfile.ZipFile(path_to_archive, "w")
    local_path_to_archive = (
        f"archive_release/{str(this_event_path_to_catalog_postfix)}/Your_archive.zip"
    )
    try:
        req = PathToArchiveRelease.objects.get(event_id=event_id)
    except:
        PathToArchiveRelease.objects.create(
            path_to_release_archive=local_path_to_archive, event_id=event_id
        )
        req = PathToArchiveRelease.objects.get(event_id=event_id)
    print(req)
    for folder, subfolders, files in os.walk(full_file_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp")):
                pass
                file_zip.write(
                    os.path.join(folder, file),
                    os.path.relpath(os.path.join(folder, file), full_save_file_path),
                    compress_type=zipfile.ZIP_DEFLATED,
                )
    return render(request, "support/download_all.html", {"path_to_archive": req})
