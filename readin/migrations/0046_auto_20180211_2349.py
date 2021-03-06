# Generated by Django 2.0 on 2018-02-11 23:49

from django.db import migrations
import multiselectfield.db.fields


class Migration(migrations.Migration):

    dependencies = [
        ('readin', '0045_auto_20180211_2323'),
    ]

    operations = [
        migrations.AlterField(
            model_name='post',
            name='method_class',
            field=multiselectfield.db.fields.MultiSelectField(blank=True, choices=[('DT', 'Decision Tree'), ('SVM', 'Support Vector Machine'), ('NN', 'Neural Network'), ('LR', 'Logistic Regression'), ('AdaC', 'Ensembled Ada Boost Classifier'), ('BC', 'Ensembled Bagging Classifier'), ('ETC', 'Ensembled Extra Trees Classifier'), ('GBC', 'Ensembled Gradient Boosting Classifier'), ('RFC', 'Ensembled Random Forest Classifier')], max_length=32),
        ),
    ]
