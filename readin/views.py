from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import ListView, TemplateView, DetailView
from django.views.generic.edit import CreateView, UpdateView, DeleteView
from .models import Post
from django.urls import reverse_lazy

# Create your views here.

class HomePageView(ListView):
	model=Post
	template_name='home.html'

class AboutPageView(TemplateView):
	template_name='about.html'

class PostDetailView(DetailView):
	model=Post
	template_name='post_detail.html'

class NewExperiment(CreateView):
	model = Post
	template_name='post_new.html'
	fields='__all__'

class EditExperiment(UpdateView):
	model=Post
	template_name='post_edit.html'
	fields='__all__'

class DeleteExperiment(DeleteView):
	model = Post
	template_name='post_delete.html'
	success_url=reverse_lazy('home')
