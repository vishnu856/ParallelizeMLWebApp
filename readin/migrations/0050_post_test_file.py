# Generated by Django 2.0 on 2018-03-19 17:20

from django.db import migrations, models
import readin.models


class Migration(migrations.Migration):

    dependencies = [
        ('readin', '0049_auto_20180305_0222'),
    ]

    operations = [
        migrations.AddField(
            model_name='post',
            name='test_file',
            field=models.FileField(blank=True, null=True, upload_to=readin.models.user_directory_test_path),
        ),
    ]
