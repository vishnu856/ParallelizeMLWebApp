from django.shortcuts import render, render_to_response, redirect
from django.template import RequestContext, loader
from django.http import HttpResponse
from django.views.generic import ListView, TemplateView, DetailView
from django.views.generic.edit import CreateView, UpdateView, DeleteView
from .models import Post
from django.core.exceptions import ValidationError
from django.urls import reverse_lazy
from django import forms
import os
import pandas as pd
from collections import defaultdict
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import utils
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
import csv
from sklearn import preprocessing
import codecs
import json
import sklearn.metrics as met

# Create your views here.

global data_file

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
		#inputfile=cleaned_data["inputfile"]
		#if not inputfile:
		#	raise ValidationError(('Upload a CSV file!'))
		#name, ext=os.path.splitext(inputfile.name)
		#if ext != ".csv":
		#	raise ValidationError(('Need to upload only .csv files!!!'))
		#data_file=pd.DataFrame(pd.read_csv(inputfile, sep=','))
		#if data_file is None:
		#	raise ValidationError(('CSV file read as empty!'))	
		if algorithm_choice:
			if algorithm_choice == 'S':
				if method_super is None or target is None:
					raise ValidationError(('Need to fill in all options for Supervised learning.'))
				#if target not in data_file:
				#	raise ValidationError(('Specify a valid target present in the CSV!'+str(target)+' '+str(data_file.columns)))
			if algorithm_choice == 'U':
				if method_unsuper is None:
					raise ValidationError(('Need to fill in all options for Unsupervised learning.'))
		else:
			raise ValidationError(('Need to select method of learning.'))						
		return cleaned_data


def encode_target(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Encode"+str(target_column)] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)

def report2dict(cr):
    # Parse rows
    tmp = list()
    for row in cr.split("\n"):
        parsed_row = [x for x in row.split("  ") if len(x) > 0]
        if len(parsed_row) > 0:
            tmp.append(parsed_row)
    
    # Store in dictionary
    measures = tmp[0]

    D_class_data = defaultdict(dict)
    for row in tmp[1:]:
        class_label = row[0]
        for j, m in enumerate(measures):
            D_class_data[class_label][m.strip()] = float(row[j + 1].strip())
    return D_class_data

def process(context, form, **kwargs):
	target=form.cleaned_data['target']
	inputfile=form.cleaned_data['inputfile']
	data_file=pd.DataFrame(pd.read_csv(inputfile, sep=','))
	#df, targets= encode_target(data_file, target)		
	X=data_file.loc[:, data_file.columns!=target]
	#X=X.loc[:, X.columns!="Encode"+str(target)]
	Y=data_file[str(target)]
#	print(Y)
	for i in X.columns:
		if isinstance(X.at[0,i],str):
			X, col=encode_target(X, i)
			X=X.loc[:, X.columns!=i]

	algorithm_choice=form.cleaned_data['algorithm_choice']
	if algorithm_choice == 'S':
		validation_split=form.cleaned_data['validation_split']
		#X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=1-(float(validation_split)/100), random_state=100)
		method_super=form.cleaned_data['method_super']
		if method_super == 'C':
			clf_gini=DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=32, min_samples_leaf=5)
			clf_gini.fit(X, Y)
			Y_pred=cross_val_predict(clf_gini, X, Y, cv=int(100-validation_split)) #clf_gini.predict(X_test)	
			X_test=X
			Y_test=Y
			context['x_cols']=X_test.columns
			context['tot_cols']=range(len(X_test.columns)+2)
			context['y_pred']=Y_pred 
			context['y_test']=Y_test 
			zip_val=list(zip(Y_test, Y_pred))
			Y_pred=np.matrix(Y_pred).T
			Y_test=np.matrix(Y_test).T
			val_set=np.concatenate([X_test, Y_test], axis=1)
			test_zip=np.concatenate([val_set, Y_pred],axis=1)
			context['full_set']=np.array(test_zip)
#			print(zip_val)
			class_report=pd.DataFrame(report2dict(met.classification_report(Y_test, Y_pred)))
			print(class_report)
			context['class_report']=class_report.to_html()
			context['zip_json']=json.dumps(zip_val)
			context['form']=form
			# here you can add things like:
			return render_to_response("result_class.html",context)


		if method_super == 'R':
			lr=LinearRegression()
			rgr_gini=DecisionTreeRegressor(criterion="mse", random_state=128, max_depth=32, min_samples_leaf=1)
#			lr.fit(X_train, Y_train)
#			print(Y)
#			Y_pred=lr.predict(X_test)
			Y_pred=cross_val_predict(rgr_gini, X, Y, cv=int(100-validation_split))
			X_test=X
			Y_test=Y
			context['x_cols']=X_test.columns
			context['tot_cols']=range(len(X_test.columns)+2)
			context['y_pred']=Y_pred 
			context['y_test']=Y_test 
			zip_val=list(zip(Y_test, Y_pred))
			Y_pred=np.matrix(Y_pred).T
			Y_test=np.matrix(Y_test).T
			val_set=np.concatenate([X_test, Y_test], axis=1)
			test_zip=np.concatenate([val_set, Y_pred],axis=1)
			context['full_set']=np.array(test_zip)
#			print(zip_val)
			context['zip_json']=json.dumps(zip_val)
			context['form']=form
			# here you can add things like:
			return render_to_response("result_reg.html",context)
		context['error']="This is an error page. You are not supposed to see this."
		return render_to_response("home.html", context)

class NewExperiment(CreateView):
	model = Post
	form_class=NewExperimentForm
	template_name='post_new.html'
	#success_url='/result/'	

	def form_invalid(self, form, **kwargs):
		context = self.get_context_data(**kwargs)
		context['form'] = form
		# here you can add things like:
		return self.render_to_response(context)

	def form_valid(self, form, **kwargs):
		context = self.get_context_data(**kwargs)
		temp=process(context, form, **kwargs)
		self.object=form.save()
		return temp


	def post(self, request, *args, **kwargs):
		self.object=None
		form_class = self.get_form_class()
		form = self.get_form(form_class)
		if form.is_valid():
			return self.form_valid(form, **kwargs)
		else:
		    return self.form_invalid(form, **kwargs)

'''
def newExperiment(request):
	if request.method== 'POST':
		form=NewExperimentForm(request.POST, request.FILES)
		if form.is_valid():			
			target=form.cleaned_data['target']
			inputfile=request.FILES['inputfile']
			if inputfile:
				Post(inputfile=inputfile).save()
				data_file=pd.DataFrame(pd.read_csv(inputfile, sep=','))
				X=data_file.loc[:, data_file.columns!=target]		
				Y=data_file.loc[:, data_file.columns==target]
				X_train, Y_train, X_test, Y_test=train_test_split(X,Y,test_size=0.3, random_state=100)
				algorithm_choice=form.cleaned_data['algorithm_choice']
				if algorithm_choice == 'S':
					method_super=form.cleaned_data['method_super']
					if method_super == 'C':
						clf_gini=DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
						clf_gini.fit(X_train, Y_train)
						Y_pred=clf_gini.predict(X_test)
						context['y_pred']=Y_pred
						context['y_test']=Y_test
				context['form']=form
				# here you can add things like:
				return render(request, "result.html",context)
	else:
		form=NewExperimentForm()
	return render(request, 'post_new.html', {'form' : form})
'''
			

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
		#inputfile=cleaned_data["inputfile"]
		#if not inputfile:
		#	raise ValidationError(('Upload a CSV file!'))
		#name, ext=os.path.splitext(inputfile.name)
		#if ext != ".csv":
		#	raise ValidationError(('Need to upload only .csv files!!!'))
		#data_file=pd.DataFrame(pd.read_csv(inputfile, sep=','))
		#if data_file is None:
		#	raise ValidationError(('CSV file read as empty!'))	
		if algorithm_choice:
			if algorithm_choice == 'S':
				if method_super is None or target is None:
					raise ValidationError(('Need to fill in all options for Supervised learning.'))
				#if target not in data_file:
				#	raise ValidationError(('Specify a valid target present in the CSV!'+str(target)+' '+str(data_file.columns)))
			if algorithm_choice == 'U':
				if method_unsuper is None:
					raise ValidationError(('Need to fill in all options for Unsupervised learning.'))
		else:
			raise ValidationError(('Need to select method of learning.'))						
		return cleaned_data

class EditExperiment(UpdateView):
	model=Post
	template_name='post_edit.html'
	form_class=EditExperimentForm

	def form_invalid(self, form, **kwargs):
		context = self.get_context_data(**kwargs)
		context['form'] = form
		# here you can add things like:
		return self.render_to_response(context)

	def form_valid(self, form, **kwargs):
		context = self.get_context_data(**kwargs)
		temp=process(context, form, **kwargs)
		self.object=form.save(commit=False)
		self.object.pk=kwargs['pk']
		self.object.save(force_update=True)
		return temp


	def post(self, request, *args, **kwargs):
		self.object=None
		form_class = self.get_form_class()
		form = self.get_form(form_class)
		if form.is_valid():
		    return self.form_valid(form, **kwargs)
		else:
		    return self.form_invalid(form, **kwargs)
	

class DeleteExperiment(DeleteView):
	model = Post
	template_name='post_delete.html'
	success_url=reverse_lazy('home')
