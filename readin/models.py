from django.db import models
from os.path import basename
from django.urls import reverse

# Create your models here.

def user_directory_path(instance, filename):
	return 'users/{0}'.format(filename)

class Post(models.Model):
	inputfile=models.FileField(upload_to=user_directory_path, null=True)
	
	def get_absolute_url(self):
		return reverse('home')

	def __str__(self):
		return basename(self.inputfile.name)
