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
import sklearn.feature_selection as FS
from sklearn.linear_model import LinearRegression, LogisticRegression, BayesianRidge
import sklearn.svm as svm
import sklearn.neural_network as NN
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.decomposition import PCA
import csv
from sklearn import preprocessing
import codecs
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, Birch
import json
import sklearn.metrics as met
from scipy import interp
from formtools.preview import FormPreview

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
				method_cl=cleaned_data.get("method_class")
				if method_super=='C' and method_cl is None:
					raise ValidationError(('Need to fill in all options for Classification.'))	
				method_re=cleaned_data.get("method_reg")
				if method_super=='R' and method_re is None:
					raise ValidationError(('Need to fill in all options for Regression.'))					
				#if target not in data_file:
				#	raise ValidationError(('Specify a valid target present in the CSV!'+str(target)+' '+str(data_file.columns)))
			if algorithm_choice == 'U':
				if method_unsuper is None:
					raise ValidationError(('Need to fill in all options for Unsupervised learning.'))
				method_clust=cleaned_data.get("method_clust")
				if method_unsuper == 'C' and method_clust is None:
					raise ValidationError(('Need to fill in all options for Clustering.'))
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
    df_mod["Encode "+str(target_column)] = df_mod[target_column].replace(map_to_int)

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

def process(data_file, context, form, **kwargs):

	X=data_file.fillna(0)

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
			method_class=form.cleaned_data['method_class']
			if method_class == 'DT':
				clf=DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=32, min_samples_leaf=5)
				#clf_gini.fit(X, Y)
			if method_class == 'SVM':
				clf=svm.LinearSVC()
			if method_class == 'NN':
				clf=NN.MLPClassifier()	
			if method_class == 'LR':
				clf=LogisticRegression()
			Y_pred=cross_val_predict(clf, X, Y, cv=int(100-validation_split)) #clf_gini.predict(X_test)
			print(Y_pred)
			X_test=data_file.loc[:, data_file.columns!=target]
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
			method_reg=form.cleaned_data['method_reg']
			if method_reg == 'LR':			
				rgr=LinearRegression()
			if method_reg == 'DT':	
				rgr=DecisionTreeRegressor(criterion="mse", random_state=128, max_depth=32, min_samples_leaf=1)
			if method_reg == 'BR':
				rgr=BayesianRidge()
			if method_reg == 'SVR':
				rgr=svm.SVR()
			Y_pred=cross_val_predict(rgr, X, Y, cv=int(100-validation_split))
			X_test=data_file.loc[:, data_file.columns!=target]
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
			method_clust=form.cleaned_data['method_clust']
			if method_clust == 'KM':
				clust=KMeans(n_clusters=form.cleaned_data['no_clusters']).fit(X)
			if method_clust == 'SC':
				clust=SpectralClustering(n_clusters=form.cleaned_data['no_clusters']).fit(X)
			if method_clust == 'AC':
				clust=AgglomerativeClustering(n_clusters=form.cleaned_data['no_clusters']).fit(X)
			if method_clust == 'BC':
				clust=Birch(n_clusters=form.cleaned_data['no_clusters']).fit(X)
			context['x_cols']=data_file.columns
			context['tot_cols']=range(len(data_file.columns)+1)
			context['y_pred']=clust.labels_
			Y_pred=np.matrix(clust.labels_).T
			tot_zip=np.concatenate([data_file, Y_pred], axis=1)
			context['full_set']=np.array(tot_zip)
			context['silhouette']=met.silhouette_score(X, labels=Y_pred)
			context['chscore']=met.calinski_harabaz_score(X, labels=Y_pred)
			context['form']=form

			pca=PCA(n_components=2).fit(X)
			pca_2d=pca.transform(X)
			#print(X.shape)
			#print(pca_2d)
			y_pred=[]
			for r in clust.labels_:
				y_pred.append("Cluster "+str(r))
			full_set=np.concatenate([pca_2d, pd.DataFrame(np.matrix(y_pred)).T], axis=1)
			#print(full_set)
			context['zip_json']=json.dumps(full_set.tolist())
			return render_to_response("result_clust.html", context)
	if algorithm_choice == 'F':
		target=form.cleaned_data['target']
		no_features=form.cleaned_data['no_features']
		X=data_file
		for i in X.columns:
			if isinstance(X.at[0,i],str):
				X, col=encode_target(X, i)
				X=X.loc[:, X.columns!=i]
		try:
			Y=X[target]
			X=X.loc[:, X.columns!=target]
		except KeyError as k:
			Y=X["Encode "+str(target)]
			X=X.loc[:, X.columns!="Encode "+str(target)]

		fs_model=FS.SelectKBest(k=no_features)
		X_new=pd.DataFrame(fs_model.fit_transform(X, Y))
		#print(X_new)
		cols=[]
		i=0
		for c in fs_model.get_support():			
			if c == True:
				print(i)
				print(data_file.columns[i+1])
				cols.append(data_file.columns[i+1])
			i=i+1
		print(cols)
		context['x_new_cols']=cols
		context['x_cols']=[d for d in data_file.columns if d!=target]
		context['full_set']=np.array(pd.DataFrame(data_file, columns=list(cols)))#np.array(X_new)
		context['scores']=fs_model.scores_
		#print(context['scores'])
		context['supports']=fs_model.get_support()
		return render_to_response("result_feature.html", context)

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
		inputfile=form.cleaned_data['inputfile']
		data_file=pd.DataFrame(pd.read_csv(inputfile, sep=',', keep_default_na=False))
		temp=process(data_file, context, form, **kwargs)
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

class NewExperimentPreview(FormPreview):
	form_template="post_new.html"
	preview_template="post_detail.html"
	
	def done(self, request, cleaned_data):
		return render_to_response("result_class.html")

class EditExperimentForm(forms.ModelForm):
	class Meta:
		model = Post
		exclude=['inputfile']
		widgets={
#			'inputfile': forms.FileInput,	
			'algorithm_choice': forms.Select(attrs={'onchange':"showDiv(this);"}),
			'method_super': forms.Select(attrs={'onchange':"showSuperDiv(this);"}),
			'method_unsuper': forms.Select(attrs={'onchange':"showUnsuperDiv(this);"}),	
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
				method_cl=cleaned_data.get("method_class")
				if method_super=='C' and method_cl is None:
					raise ValidationError(('Need to fill in all options for Classification.'))	
				method_re=cleaned_data.get("method_reg")
				if method_super=='R' and method_re is None:
					raise ValidationError(('Need to fill in all options for Regression.'))					
				#if target not in data_file:
				#	raise ValidationError(('Specify a valid target present in the CSV!'+str(target)+' '+str(data_file.columns)))
			if algorithm_choice == 'U':
				if method_unsuper is None:
					raise ValidationError(('Need to fill in all options for Unsupervised learning.'))
				method_clust=cleaned_data.get("method_clust")
				if method_unsuper == 'C' and method_clust is None:
					raise ValidationError(('Need to fill in all options for Clustering.'))

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
		p=Post.objects.get(pk=context['pk'])
		data_file=p.get_inputfile_as_DF()
		#print(form.cleaned_data["no_features"])
		temp=process(data_file, context, form, **kwargs)
		self.object=form.save(commit=False)
		self.object.pk=kwargs['pk']
		self.object.save(update_fields=[f.name for f in Post._meta.get_fields() if f.name !='inputfile' and f.name != 'id'], force_update=True)
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
