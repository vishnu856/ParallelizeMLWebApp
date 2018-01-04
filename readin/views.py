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
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.decomposition import PCA
import csv
from sklearn import preprocessing
import codecs
from sklearn.cluster import KMeans
import json
import sklearn.metrics as met
from scipy import interp
import matplotlib.pyplot as pl, mpld3

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
	inputfile=form.cleaned_data['inputfile']
	data_file=pd.DataFrame(pd.read_csv(inputfile, sep=','))
	#df, targets= encode_target(data_file, target)		
	#X=X.loc[:, X.columns!="Encode"+str(target)]
#	print(Y)
	X=data_file

	algorithm_choice=form.cleaned_data['algorithm_choice']
	if algorithm_choice == 'S':
		target=form.cleaned_data['target']
		X=data_file.loc[:, data_file.columns!=target]
		for i in X.columns:
			if isinstance(X.at[0,i],str):
				X, col=encode_target(X, i)
				X=X.loc[:, X.columns!=i]

		Y=data_file[str(target)]
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
			context['class_report']=class_report.to_html()
			context['classes']=list(set(Y))
			lookup={}
			i=0
			for c in context['classes']:
				lookup[i]=c
				i=i+1
			conf_mat=pd.DataFrame(met.confusion_matrix(Y_test, Y_pred, labels=context['classes'])).rename(lookup, axis='index').rename(lookup, axis='columns')
			context['confusion_matrix']=conf_mat.to_html()
			prec, recall, fscore, support=met.precision_recall_fscore_support(Y_test, Y_pred, average="macro")
			context['precision']=prec
			context['recall']=recall
			context['fscore']=fscore
			context['accuracy']=met.accuracy_score(Y_test, Y_pred)
			#print(Y)
			bin_y=preprocessing.label_binarize(Y, classes=context['classes'])
			bin_y_test=preprocessing.label_binarize(Y_test, classes=context['classes'])			
			bin_y_pred=preprocessing.label_binarize(Y_pred, classes=context['classes'])			
			n_classes=bin_y.shape[1]
			fpr=dict()
			tpr=dict()
			roc_auc = dict()

			print(n_classes)
			for i in range(n_classes):
				fpr[i], tpr[i], _ = met.roc_curve(bin_y_test[:, i], bin_y_pred[:, i])
				roc_auc[i] = met.auc(fpr[i], tpr[i])
			all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
			mean_tpr = np.zeros_like(all_fpr)
			for i in range(n_classes):
			    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
			mean_tpr /= n_classes
			context['fpr']=all_fpr
			context['tpr']=mean_tpr
			context['roc_auc']=met.auc(all_fpr, mean_tpr)
			print(all_fpr)
			print(mean_tpr)			

			context['zip_json']=json.dumps(list(zip(all_fpr, mean_tpr)))
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
			context['explained_variance_score']=met.explained_variance_score(Y_test, Y_pred)
			context['mean_absolute_error']=met.mean_absolute_error(Y_test, Y_pred)
			context['mean_squared_error']=met.mean_squared_error(Y_test, Y_pred)
			context['mean_squared_log_error']=met.mean_squared_log_error(Y_test, Y_pred)
			context['median_absolute_error']=met.median_absolute_error(Y_test, Y_pred)
			context['r2_score']=met.r2_score(Y_test, Y_pred)
#			print(zip_val)
			context['zip_json']=json.dumps(zip_val)
			context['form']=form
			# here you can add things like:
			return render_to_response("result_reg.html",context)
	if algorithm_choice == 'U':
		method_unsuper=form.cleaned_data['method_unsuper']
		for i in X.columns:
			if isinstance(X.at[0,i],str):
				X, col=encode_target(X, i)
				X=X.loc[:, X.columns!=i]
		if method_unsuper=='C':
			kmeans=KMeans(n_clusters=form.cleaned_data['no_clusters']).fit(X)
			context['x_cols']=data_file.columns
			context['tot_cols']=range(len(data_file.columns)+1)
			context['y_pred']=kmeans.labels_
			Y_pred=np.matrix(kmeans.labels_).T
			tot_zip=np.concatenate([data_file, Y_pred], axis=1)
			context['full_set']=np.array(tot_zip)
			context['silhouette']=met.silhouette_score(X, labels=Y_pred)
			context['chscore']=met.calinski_harabaz_score(X, labels=Y_pred)
			context['form']=form

			pca=PCA(n_components=2).fit(X)
			pca_2d=pca.transform(X)
			print(X.shape)
			print(pca_2d)
			y_pred=[]
			for r in kmeans.labels_:
				y_pred.append("Cluster "+str(r))
			full_set=np.concatenate([pca_2d, pd.DataFrame(np.matrix(y_pred)).T], axis=1)
			print(full_set)
			context['zip_json']=json.dumps(full_set.tolist())
			return render_to_response("result_clust.html", context)
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
