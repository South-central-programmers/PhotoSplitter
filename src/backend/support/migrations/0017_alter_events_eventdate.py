# Generated by Django 5.0.1 on 2024-01-26 18:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('support', '0016_remove_events_creator_id_events_user_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='events',
            name='eventdate',
            field=models.CharField(max_length=30),
        ),
    ]
