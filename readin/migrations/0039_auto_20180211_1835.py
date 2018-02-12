# Generated by Django 2.0 on 2018-02-11 18:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('readin', '0038_auto_20180211_1823'),
    ]

    operations = [
        migrations.RenameField(
            model_name='post',
            old_name='is_hyper',
            new_name='is_class_hyper',
        ),
        migrations.AddField(
            model_name='post',
            name='is_reg_hyper',
            field=models.CharField(blank=True, choices=[('Y', 'Yes'), ('N', 'No')], max_length=128),
        ),
    ]