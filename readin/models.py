from django.db import models
from os.path import basename
from django.urls import reverse
from django import forms
import pandas as pd

# Create your models here.

def user_directory_path(instance, filename):
	return 'users/{0}'.format(filename)

class Post(models.Model):
	ALGO_CHOICES=(
		('S','Supervised'),
		('U','Unsupervised'),
		('F', 'Feature Selection'),
	)
	ALGO_SUPER_CHOICES=(
		('C', 'Classification'),
		('R', 'Regression'),	
	)
	ALGO_CLASS_CHOICES=(
		('DT','Decision Tree'),
		('SVM','Support Vector Machine'),
		('NN','Neural Network'),		
		('LR','Logistic Regression'),
	)
	ALGO_UNSUPER_CHOICES=(
		('C', 'Clustering'),
	)
	ALGO_REG_CHOICES=(
		('DT', 'Decision Tree'),
		('BR', 'Bayesian Ridge'),
		('SVR', 'Support Vector Regression'),
		('LR', 'Linear Regression'),
	)
	ALGO_CLUST_CHOICES=(
		('KM', 'K-Means Clustering'),
		('SC', 'Spectral Clustering'),
		('AC', 'Agglomerative Clustering'),
		('BC', 'Birch Clustering'),
	)
	ALGO_HYPERPARAM_TUNING=(
		('GR', 'Grid Search CV'),
		('RR', 'Randomized Search CV'),	
	)
	CHOICE_HYPERPARAM_TUNING=(
		('Y', 'Yes'),
		('N', 'No'),
	)
	title=models.CharField(max_length=200, default="Untitled")
	inputfile=models.FileField(upload_to=user_directory_path, null=True, blank=True)	
	comments=models.CharField(max_length=500, blank=True, null=True)
	algorithm_choice=models.CharField(max_length=2, choices=ALGO_CHOICES, blank=True)
	
	validation_split=models.DecimalField(max_digits=3, decimal_places=1, default=75)
	method_super=models.CharField(max_length=128, choices=ALGO_SUPER_CHOICES, blank=True)	
	no_labels=models.PositiveSmallIntegerField(default=2)

	method_class=models.CharField(max_length=128, choices=ALGO_CLASS_CHOICES, blank=True)	
	method_reg=models.CharField(max_length=128, choices=ALGO_REG_CHOICES, blank=True)	

	is_hyper=models.CharField(max_length=128, choices=CHOICE_HYPERPARAM_TUNING, blank=True)
	method_hyper=models.CharField(max_length=128, choices=ALGO_HYPERPARAM_TUNING, blank=True)	
	
	target=models.CharField(max_length=500, blank=True, null=True)

	method_unsuper=models.CharField(max_length=128, choices=ALGO_UNSUPER_CHOICES, blank=True)

	no_clusters=models.PositiveSmallIntegerField(default=2)
	method_clust=models.CharField(max_length=128, choices=ALGO_CLUST_CHOICES, blank=True)	

	no_features=models.PositiveSmallIntegerField(default=2)

	def get_absolute_url(self):
		return reverse('home')

	def __str__(self):
		return basename(self.title)

	def get_inputfile_text(self):
		return pd.DataFrame(pd.read_csv(self.inputfile, sep=',')).to_html(classes="highlight centered")
	
	def get_inputfile_as_DF(self):
		return pd.DataFrame(pd.read_csv(self.inputfile, sep=',', keep_default_na=False))
