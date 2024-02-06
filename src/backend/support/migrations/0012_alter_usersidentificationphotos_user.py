# Generated by Django 5.0.1 on 2024-01-26 16:49

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("support", "0011_remove_usersidentificationphotos_id_of_user_and_more"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AlterField(
            model_name="usersidentificationphotos",
            name="user",
            field=models.ForeignKey(
                default=None,
                on_delete=django.db.models.deletion.SET_DEFAULT,
                to=settings.AUTH_USER_MODEL,
            ),
        ),
    ]
