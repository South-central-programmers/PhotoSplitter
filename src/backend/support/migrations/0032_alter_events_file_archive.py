# Generated by Django 5.0.1 on 2024-01-31 18:15

import django.core.validators
import support.models
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("support", "0031_alter_events_file_archive"),
    ]

    operations = [
        migrations.AlterField(
            model_name="events",
            name="file_archive",
            field=models.FileField(
                upload_to=support.models.events_path,
                validators=[
                    django.core.validators.FileExtensionValidator(
                        allowed_extensions=["zip", "rar", "gz"]
                    )
                ],
            ),
        ),
    ]
