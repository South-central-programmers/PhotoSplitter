# Generated by Django 5.0.1 on 2024-01-29 22:18

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('support', '0021_eventsheadbands'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='eventsheadbands',
            name='event_id',
        ),
        migrations.AddField(
            model_name='eventsheadbands',
            name='event',
            field=models.ForeignKey(default=None, on_delete=django.db.models.deletion.CASCADE, to='support.events'),
        ),
    ]
