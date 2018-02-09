# Generated by Django 2.0 on 2018-02-09 21:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('readin', '0031_remove_post_method_hyper'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='post',
            name='method_ensemble',
        ),
        migrations.AddField(
            model_name='post',
            name='method_class_ensemble',
            field=models.CharField(blank=True, choices=[('AdaC', 'Ada Boost Classifier')], max_length=128),
        ),
    ]
