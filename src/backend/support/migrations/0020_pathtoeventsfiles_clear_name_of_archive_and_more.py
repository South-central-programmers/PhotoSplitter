# Generated by Django 5.0.1 on 2024-01-28 21:32

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("support", "0019_pathtoeventsfiles"),
    ]

    operations = [
        migrations.AddField(
            model_name="pathtoeventsfiles",
            name="clear_name_of_archive",
            field=models.CharField(default=None, max_length=150),
        ),
        migrations.AddField(
            model_name="pathtoeventsfiles",
            name="type_of_archive",
            field=models.CharField(default=None, max_length=150),
        ),
    ]