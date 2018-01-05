# Generated by Django 2.0 on 2018-01-05 19:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('readin', '0025_auto_20180105_1840'),
    ]

    operations = [
        migrations.AddField(
            model_name='post',
            name='method_clust',
            field=models.CharField(blank=True, choices=[('KM', 'K-Means Clustering'), ('SC', 'Spectral Clustering'), ('DB', 'DBSCAN Clustering'), ('AC', 'Agglomerative Clustering'), ('BC', 'Birch Clustering')], max_length=128),
        ),
    ]