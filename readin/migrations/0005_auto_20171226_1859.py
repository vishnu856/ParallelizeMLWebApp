# Generated by Django 2.0 on 2017-12-26 18:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('readin', '0004_auto_20171226_1857'),
    ]

    operations = [
        migrations.AlterField(
            model_name='post',
            name='algorithm_choice',
            field=models.CharField(choices=[('Supervised', 'S'), ('Unsupervised', 'U')], default='Supervised', max_length=128),
        ),
    ]