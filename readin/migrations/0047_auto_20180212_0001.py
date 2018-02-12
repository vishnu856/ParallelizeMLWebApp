# Generated by Django 2.0 on 2018-02-12 00:01

from django.db import migrations
import multiselectfield.db.fields


class Migration(migrations.Migration):

    dependencies = [
        ('readin', '0046_auto_20180211_2349'),
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
            field=multiselectfield.db.fields.MultiSelectField(blank=True, choices=[('DT', 'Decision Tree'), ('BR', 'Bayesian Ridge'), ('SVR', 'Support Vector Regression'), ('LR', 'Linear Regression'), ('AdaR', 'Ada Boost Regressor'), ('BR', 'Bagging Regressor'), ('ETR', 'Extra Trees Regressor'), ('GBR', 'Gradient Boosting Regressor'), ('RFR', 'Random Forest Regressor')], max_length=32),
        ),
    ]
