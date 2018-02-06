#readin/urls.py

from django.urls import path
from . import views

urlpatterns = [
	path('', views.HomePageView.as_view(), name='home'),
	path('about/', views.AboutPageView.as_view(), name='about'),
	path('result/', views.AboutPageView.as_view(), name='about'),
#	path('post/<int:pk>/', views.DetailExperiment.as_view(), name='post_detail'),
	path('post/new/', views.NewExperiment.as_view(), name='new_exp'),
#	path('post/new/', views.NewExperimentPreview(views.NewExperimentForm), name='new_exp'),
	#path('post/new/', views.newExperiment, name='new_exp'),
	path('post/<int:pk>/edit/', views.EditExperiment.as_view(), name='post_edit'),
	path('post/<int:pk>/delete/', views.DeleteExperiment.as_view(), name='post_delete'),
]
