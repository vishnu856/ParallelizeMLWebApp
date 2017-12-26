from django.db import models
from os.path import basename
from django.urls import reverse

# Create your models here.

def user_directory_path(instance, filename):
	return 'users/{0}'.format(filename)

class Post(models.Model):
	ALGO_CHOICES=(
		('S','Supervised'),
		('U','Unsupervised'),
	)
	title=models.CharField(max_length=200, default="Untitled")
	inputfile=models.FileField(upload_to=user_directory_path, null=True)
	comments=models.CharField(max_length=500, null=True)
	algorithm_choice=models.CharField(max_length=2, choices=ALGO_CHOICES, default='S')

	def get_absolute_url(self):
		return reverse('home')

	def __str__(self):
		return basename(self.inputfile.title)
