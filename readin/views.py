from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import ListView, TemplateView, DetailView
from django.views.generic.edit import CreateView, UpdateView, DeleteView
from .models import Post
from django.core.exceptions import ValidationError
from django.urls import reverse_lazy
from django import forms
import os

# Create your views here.

class HomePageView(ListView):
	model=Post
	template_name='home.html'

class AboutPageView(TemplateView):
	template_name='about.html'

class NewExperimentForm(forms.ModelForm):
	class Meta:
		model = Post
		fields='__all__'
		widgets={
			'algorithm_choice': forms.Select(attrs={'onchange':"showDiv(this);"}),
			'method_super': forms.Select(attrs={'onchange':"showSuperDiv(this);"}),
			'method_unsuper': forms.Select(attrs={'onchange':"showUnsuperDiv(this);"}),
		}
	def clean(self):
		cleaned_data=super(NewExperimentForm, self).clean()
		algorithm_choice=cleaned_data.get("algorithm_choice")
		method_super=cleaned_data.get("method_super")
		method_unsuper=cleaned_data.get("method_unsuper")
		target=cleaned_data.get("target")		
		if algorithm_choice:
			if algorithm_choice == 'S':
				if method_super is None or target is None:
					raise ValidationError(('Need to fill in all options for Supervised learning.'))
			if algorithm_choice == 'U':
				if method_unsuper is None:
					raise ValidationError(('Need to fill in all options for Unsupervised learning.'))
		else:
			raise ValidationError(('Need to select method of learning.'))		
		inputfile=cleaned_data.get("inputfile")
		name, ext=os.path.splitext(inputfile.name)
		if ext != "csv":
			raise ValidationError(('Need to upload only .csv files!!!'))
		return cleaned_data

class NewExperiment(CreateView):
	model = Post
	form_class=NewExperimentForm
	template_name='post_new.html'


class EditExperimentForm(forms.ModelForm):
	class Meta:
		model = Post
		fields='__all__'
		widgets={
			'inputfile': forms.FileInput,	
		}
	def clean(self):
		cleaned_data=super(EditExperimentForm, self).clean()
		algorithm_choice=cleaned_data.get("algorithm_choice")
		method_super=cleaned_data.get("method_super")
		method_unsuper=cleaned_data.get("method_unsuper")
		target=cleaned_data.get("target")		
		if algorithm_choice:
			if algorithm_choice == 'S':
				if method_super is None or target is None:
					raise ValidationError(('Need to fill in all options for Supervised learning.'))
			if algorithm_choice == 'U':
				if method_unsuper is None:
					raise ValidationError(('Need to fill in all options for Unsupervised learning.'))
		else:
			raise ValidationError(('Need to select method of learning.'))		
		inputfile=cleaned_data.get("inputfile")
		name, ext=os.path.splitext(inputfile.name)
		if ext != "csv":
			raise ValidationError(('Need to upload only .csv files!!!'))
		return cleaned_data

class EditExperiment(UpdateView):
	model=Post
	template_name='post_edit.html'
	success_url=reverse_lazy('home')
	form_class=EditExperimentForm
	

class DeleteExperiment(DeleteView):
	model = Post
	template_name='post_delete.html'
	success_url=reverse_lazy('home')
