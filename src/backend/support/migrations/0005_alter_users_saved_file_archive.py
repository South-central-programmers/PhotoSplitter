# Generated by Django 5.0.1 on 2024-01-25 22:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('support', '0004_users_email'),
    ]

    operations = [
        migrations.AlterField(
            model_name='users',
            name='saved_file_archive',
            field=models.FileField(blank=True, upload_to='uploads_files'),
        ),
    ]
