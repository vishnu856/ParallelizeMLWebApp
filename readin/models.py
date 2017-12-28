from django.db import models
from os.path import basename
from django.urls import reverse
from django import forms

# Create your models here.

def user_directory_path(instance, filename):
	return 'users/{0}'.format(filename)

class Post(models.Model):
	ALGO_CHOICES=(
		('S','Supervised'),
		('U','Unsupervised'),
	)
	ALGO_SUPER_CHOICES=(
		('C', 'Classification'),
		('R', 'Regression'),	
	)
	ALGO_UNSUPER_CHOICES=(
		('C', 'Clustering'),
		('F', 'Feature Selection'),
	)
	title=models.CharField(max_length=200, default="Untitled")
	inputfile=models.FileField(upload_to=user_directory_path, null=True)
	comments=models.CharField(max_length=500, blank=True, null=True)
	algorithm_choice=models.CharField(max_length=2, choices=ALGO_CHOICES, blank=True)
	
	validation_split=models.DecimalField(max_digits=3, decimal_places=1, default=75)
	method_super=models.CharField(max_length=128, choices=ALGO_SUPER_CHOICES, blank=True)	
	no_labels=models.PositiveSmallIntegerField(default=2)
	target=models.CharField(max_length=500, blank=True, null=True)

	method_unsuper=models.CharField(max_length=128, choices=ALGO_UNSUPER_CHOICES, blank=True)
	no_features=models.PositiveSmallIntegerField(default=2)
	no_clusters=models.PositiveSmallIntegerField(default=2)

	def get_absolute_url(self):
		return reverse('home')

	def __str__(self):
		return basename(self.title)
