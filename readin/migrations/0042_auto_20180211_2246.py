# Generated by Django 2.0 on 2018-02-11 22:46

from django.db import migrations
import multiselectfield.db.fields


class Migration(migrations.Migration):

    dependencies = [
        ('readin', '0041_auto_20180211_2236'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='post',
            name='is_reg_ensemble',
        ),
        migrations.RemoveField(
            model_name='post',
            name='method_reg_ensemble',
        ),
        migrations.AlterField(
            model_name='post',
            name='method_reg',
            field=multiselectfield.db.fields.MultiSelectField(choices=[('DT', 'Decision Tree'), ('BR', 'Bayesian Ridge'), ('SVR', 'Support Vector Regression'), ('LR', 'Linear Regression')], max_length=12),
        ),
    ]
