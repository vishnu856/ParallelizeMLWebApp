# Generated by Django 2.0 on 2018-02-09 22:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('readin', '0035_post_method_reg_ensemble'),
    ]

    operations = [
        migrations.RenameField(
            model_name='post',
            old_name='is_ensemble',
            new_name='is_class_ensemble',
        ),
        migrations.AddField(
            model_name='post',
            name='is_reg_ensemble',
            field=models.CharField(blank=True, choices=[('Y', 'Yes'), ('N', 'No')], max_length=128),
        ),
    ]
