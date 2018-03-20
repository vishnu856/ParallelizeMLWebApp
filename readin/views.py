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
from sklearn.model_selection import cross_val_predict, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA
import csv
from sklearn import preprocessing
import codecs
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, Birch
import json
import sklearn.metrics as met
from scipy import interp
import sklearn.ensemble as ensemble
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
			'is_reg_ensemble': forms.Select(attrs={'onchange':"showEnsembleRegDiv(this);"}),
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
				print(method_super)
				if method_super is None or target is None:
					raise ValidationError(('Need to fill in all options for Supervised learning.'))
				if method_super == 'C':
					method_cl=cleaned_data.get("method_class")
					if method_cl is None:
						raise ValidationError(('Need to select atleast 1 model'))
					if len(method_cl) > 2:
						raise ValidationError(('Can\'t select more than 2 models.'))
				if method_super == 'R':
					method_re=cleaned_data.get("method_reg")
					if method_re is None:
						raise ValidationError(('Need to select atleast 1 model'))
					if len(method_re) > 2:
						raise ValidationError(('Can\'t select more than 2 models.'))				
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

def mapping_class(s):
	clf=None
	context_str=""
	context_img=""
	params={}
	if s == 'DT':
		clf=DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=32, min_samples_leaf=5)
		context_str="Decision Tree"
		params={'random_state':np.arange(1,100,5), 'max_depth': np.arange(1,31,2), 'min_samples_leaf': np.arange(1,10,2)}
		#clf_gini.fit(X, Y)
	if s == 'SVM':
		clf=svm.LinearSVC()
		context_str="Support Vector Machine"
		params={'random_state':np.arange(1,100,5), 'C':np.arange(0.1, 1, 0.1)}
	if s == 'NN':
		clf=NN.MLPClassifier()	
		context_str="Neural Networks (Multi-layer Perceptron)"
		params={'random_state':np.arange(1,100,5), 'hidden_layer_sizes':np.arange(50,100,2), 'alpha':np.arange(0.1,1,0.1), 'max_iter':np.arange(90, 100, 1)}
	if s == 'LR':
		clf=LogisticRegression()
		context_str="Logistic Regression"
		params={'random_state':np.arange(1,100,5), 'C':np.arange(0.1, 1, 0.1), 'max_iter':np.arange(90, 100, 1)}
	if s == 'AdaC':
		clf=ensemble.AdaBoostClassifier()
		context_str="Ensemble Ada Boost Classifier"
		params={'n_estimators':np.arange(25,75), 'learning_rate':np.arange(0.1, 1, 0.1), 'random_state':np.arange(1, 100, 5)}
	if s == 'BC':
		clf=ensemble.BaggingClassifier()
		context_str="Ensemble Bagging Classifier"
		params={'n_estimators':np.arange(25,75), 'max_samples':np.arange(0.1, 1, 0.1), 'random_state':np.arange(1,100,5)}

	if s == 'ETC':
		clf=ensemble.ExtraTreesClassifier()
		context_str="Ensemble Extra Trees Classifier"
		params={'n_estimators':np.arange(25,75), 'max_depth': np.arange(1,31,2), 'min_samples_leaf': np.arange(1,10,2), 'max_features':np.arange(0.1, 1, 1), 'random_state':np.arange(1,100,5)}

	if s == 'GBC':
		clf=ensemble.GradientBoostingClassifier()
		context_str="Ensemble Gradient Boosting Classifier"												
		params={'n_estimators':np.arange(25,75), 'learning_rate':np.arange(0.1, 1, 0.1), 'max_features':np.arange(0.1, 1, 1), 'random_state':np.arange(1,100,5)}

	if s == 'RFC':
		clf=ensemble.RandomForestClassifier()
		context_str="Ensemble Random Forest Classifier"
		params={'n_estimators':np.arange(25,75), 'max_depth': np.arange(1,31,2), 'min_samples_leaf': np.arange(1,10,2) , 'max_features':np.arange(0.1, 1, 1), 'random_state':np.arange(1,100,5)}

	return clf, context_str, params, context_img

def one_model_class_render(form, data_file, Y, context, test_file, Y_pred, Y_pred_test, target):
	X_test=test_file.loc[:, test_file.columns!=target]

	X_train=data_file.loc[:, data_file.columns!=target]
	Y_train=Y	
	
	context['x_test_cols']=X_test.columns
	context['tot_test_cols']=range(len(X_train.columns)+1)
	context['y_pred_test']=Y_pred_test
	Y_test_pred=np.matrix(Y_pred_test).T
	test_val_set=np.concatenate([X_test, Y_test_pred], axis=1)
	test_zip=np.concatenate([X_test, Y_test_pred],axis=1)
	context['full_test_set']=np.array(test_zip)

	context['x_cols']=X_train.columns
	context['tot_cols']=range(len(X_train.columns)+2)
	context['y_pred']=Y_pred 
	context['y_train']=Y_train 
	zip_val=list(zip(Y_train, Y_pred))
	Y_pred=np.matrix(Y_pred).T
	Y_train=np.matrix(Y_train).T
	val_set=np.concatenate([X_train, Y_train], axis=1)
	test_zip=np.concatenate([val_set, Y_pred],axis=1)
	context['full_set']=np.array(test_zip)
#			print(zip_val)
	class_report=pd.DataFrame(report2dict(met.classification_report(Y_train, Y_pred)))
	context['class_report']=class_report.to_html()
	context['classes']=list(set(Y))
	lookup={}
	i=0
	for c in context['classes']:
		lookup[i]=c
		i=i+1
	conf_mat=pd.DataFrame(met.confusion_matrix(Y_train, Y_pred, labels=context['classes'])).rename(lookup, axis='index').rename(lookup, axis='columns')
	context['confusion_matrix']=conf_mat.to_html()
	prec, recall, fscore, support=met.precision_recall_fscore_support(Y_train, Y_pred, average="macro")
	context['precision']=prec
	context['recall']=recall
	context['fscore']=fscore
	context['accuracy']=met.accuracy_score(Y_train, Y_pred)
	#print(Y)
	bin_y=preprocessing.label_binarize(Y_train, classes=context['classes'])
	bin_y_test=preprocessing.label_binarize(Y_pred_test, classes=context['classes'])			
	bin_y_pred=preprocessing.label_binarize(Y_pred, classes=context['classes'])			
	n_classes=bin_y.shape[1]
	fpr=dict()
	tpr=dict()
	roc_auc = dict()

	print(n_classes)
	for i in range(n_classes):
		fpr[i], tpr[i], _ = met.roc_curve(bin_y[:, i], bin_y_pred[:, i])
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

def two_model_class_render(form, data_file, Y, context, test_file, Y_pred_1, Y_pred_test_1, Y_pred_2, Y_pred_test_2, target):
	X_train=data_file.loc[:, data_file.columns!=target]
	Y_train=Y

	X_test=test_file

	context['x_test_cols_1']=X_test.columns
	context['tot_test_cols_1']=range(len(X_train.columns)+1)
	context['y_pred_test_1']=Y_pred_test_1
	Y_test_pred_1=np.matrix(Y_pred_test_1).T
	test_val_set=np.concatenate([X_test, Y_test_pred_1], axis=1)
	test_zip=np.concatenate([X_test, Y_test_pred_1],axis=1)
	context['full_test_set_1']=np.array(test_zip)

	context['x_cols_1']=X_train.columns
	context['tot_cols_1']=range(len(X_train.columns)+2)
	context['y_pred_1']=Y_pred_1
	context['y_train_1']=Y_train 
	zip_val=list(zip(Y_train, Y_pred_1))
	Y_pred_1=np.matrix(Y_pred_1).T
	Y_train=np.matrix(Y_train).T
	val_set=np.concatenate([X_train, Y_train], axis=1)
	print(val_set.shape)
	print(Y_pred_1.shape)
	test_zip=np.concatenate([val_set, Y_pred_1],axis=1)
	context['full_set_1']=np.array(test_zip)
#			print(zip_val)
	class_report=pd.DataFrame(report2dict(met.classification_report(Y_train, Y_pred_1)))
	context['class_report_1']=class_report.to_html()
	context['classes']=list(set(Y))
	lookup={}
	i=0
	for c in context['classes']:
		lookup[i]=c
		i=i+1
	conf_mat=pd.DataFrame(met.confusion_matrix(Y_train, Y_pred_1, labels=context['classes'])).rename(lookup, axis='index').rename(lookup, axis='columns')
	context['confusion_matrix_1']=conf_mat.to_html()
	prec, recall, fscore, support=met.precision_recall_fscore_support(Y_train, Y_pred_1, average="macro")
	context['precision_1']=prec
	context['recall_1']=recall
	context['fscore_1']=fscore
	context['accuracy_1']=met.accuracy_score(Y_train, Y_pred_1)
	#print(Y)
	bin_y=preprocessing.label_binarize(Y, classes=context['classes'])
	bin_y_test=preprocessing.label_binarize(Y_pred_test_1, classes=context['classes'])			
	bin_y_pred=preprocessing.label_binarize(Y_pred_1, classes=context['classes'])			
	n_classes=bin_y.shape[1]
	fpr=dict()
	tpr=dict()
	roc_auc = dict()

	print(n_classes)
	for i in range(n_classes):
		fpr[i], tpr[i], _ = met.roc_curve(bin_y[:, i], bin_y_pred[:, i])
		roc_auc[i] = met.auc(fpr[i], tpr[i])
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(n_classes):
	    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
	mean_tpr /= n_classes
	context['fpr_1']=all_fpr
	context['tpr_1']=mean_tpr
	context['roc_auc_1']=met.auc(all_fpr, mean_tpr)
	print(all_fpr)
	print(mean_tpr)			

	context['zip_json_1']=json.dumps(list(zip(all_fpr, mean_tpr)))
	context['form']=form



	X_train=data_file.loc[:, data_file.columns!=target]
	Y_train=Y

	X_test=test_file

	context['x_test_cols_2']=X_test.columns
	context['tot_test_cols_2']=range(len(X_train.columns)+1)
	context['y_pred_test_2']=Y_pred_test_2
	Y_test_pred_2=np.matrix(Y_pred_test_2).T
	test_val_set=np.concatenate([X_test, Y_test_pred_2], axis=1)
	test_zip=np.concatenate([X_test, Y_test_pred_2],axis=1)
	context['full_test_set_2']=np.array(test_zip)

	context['x_cols_2']=X_train.columns
	context['tot_cols_2']=range(len(X_train.columns)+2)
	context['y_pred_2']=Y_pred_2
	context['y_train_2']=Y_train 
	zip_val=list(zip(Y_train, Y_pred_2))
	Y_pred_2=np.matrix(Y_pred_2).T
	Y_train=np.matrix(Y_train).T
	val_set=np.concatenate([X_train, Y_train], axis=1)

	test_zip=np.concatenate([val_set, Y_pred_2],axis=1)
	context['full_set_2']=np.array(test_zip)
#			print(zip_val)
	class_report=pd.DataFrame(report2dict(met.classification_report(Y_train, Y_pred_2)))
	context['class_report_2']=class_report.to_html()
	context['classes']=list(set(Y))
	lookup={}
	i=0
	for c in context['classes']:
		lookup[i]=c
		i=i+1
	conf_mat=pd.DataFrame(met.confusion_matrix(Y_train, Y_pred_2, labels=context['classes'])).rename(lookup, axis='index').rename(lookup, axis='columns')
	context['confusion_matrix_2']=conf_mat.to_html()
	prec, recall, fscore, support=met.precision_recall_fscore_support(Y_train, Y_pred_2, average="macro")
	context['precision_2']=prec
	context['recall_2']=recall
	context['fscore_2']=fscore
	context['accuracy_2']=met.accuracy_score(Y_train, Y_pred_2)
	#print(Y)
	bin_y=preprocessing.label_binarize(Y, classes=context['classes'])
	bin_y_test=preprocessing.label_binarize(Y_pred_test_2, classes=context['classes'])			
	bin_y_pred=preprocessing.label_binarize(Y_pred_2, classes=context['classes'])			
	n_classes=bin_y.shape[1]
	fpr=dict()
	tpr=dict()
	roc_auc = dict()

	print(n_classes)
	for i in range(n_classes):
		fpr[i], tpr[i], _ = met.roc_curve(bin_y[:, i], bin_y_pred[:, i])
		roc_auc[i] = met.auc(fpr[i], tpr[i])
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(n_classes):
	    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
	mean_tpr /= n_classes
	context['fpr_2']=all_fpr
	context['tpr_2']=mean_tpr
	context['roc_auc_2']=met.auc(all_fpr, mean_tpr)
	print(all_fpr)
	print(mean_tpr)			

	context['zip_json_2']=json.dumps(list(zip(all_fpr, mean_tpr)))
	return render_to_response("result_class_comp.html", context)

def mapping_reg(s):
	rgr=None
	context_str=""
	context_img=""
	params={}
	if s == 'LR':			
		rgr=LinearRegression()
		context_str="Linear Regressor"
		params={'n_jobs': np.arange(1,11,1)}

	if s == 'DT':	
		rgr=DecisionTreeRegressor(criterion="mse", random_state=128, max_depth=32, min_samples_leaf=1)
		context_str="Decision Tree Regressor"
		params={'random_state':np.arange(1,100,5), 'max_depth': np.arange(1,31,2), 'min_samples_leaf': np.arange(1,10,2)}

	if s == 'BayR':
		rgr=BayesianRidge()
		context_str="Bayesian Ridge Regressor"
		params={'lambda_1':np.arange(1,100,5), 'n_iter': np.arange(1,31,2), 'alpha_1': np.arange(1,10,2)}

	if s == 'SVR':
		rgr=svm.SVR()
		context_str="Support Vector Regressor"
		params={'max_iter':np.arange(1,100,5), 'C':np.arange(0.1, 1, 0.1)}
	
	if s == 'AdaR':
		rgr=ensemble.AdaBoostRegressor()
		context_str="Ensemble Ada Boost Regressor"
		params={'n_estimators':np.arange(25,75), 'learning_rate':np.arange(0.1, 1, 0.1), 'random_state':np.arange(1, 100, 5)}

	if s == 'BagR':
		rgr=ensemble.BaggingRegressor()
		context_str="Ensemble Bagging Regressor"
		params={'n_estimators':np.arange(25,75), 'max_samples':np.arange(0.1, 1, 0.1), 'random_state':np.arange(1,100,5)}

	if s == 'ETR':
		rgr=ensemble.ExtraTreesRegressor()
		context_str="Ensemble Extra Trees Regressor"
		params={'n_estimators':np.arange(25,75), 'max_depth': np.arange(1,31,2), 'min_samples_leaf': np.arange(1,10,2), 'max_features':np.arange(0.1, 1, 1), 'random_state':np.arange(1,100,5)}

	if s == 'GBR':
		rgr=ensemble.GradientBoostingRegressor()
		context_str="Ensemble Gradient Boosting Regressor"					
		params={'n_estimators':np.arange(25,75), 'learning_rate':np.arange(0.1, 1, 0.1), 'max_features':np.arange(0.1, 1, 1), 'random_state':np.arange(1,100,5)}
							
	if s == 'RFR':
		rgr=ensemble.RandomForestRegressor()
		context_str="Ensemble Random Forest Regressor"
		params={'n_estimators':np.arange(25,75), 'max_depth': np.arange(1,31,2), 'min_samples_leaf': np.arange(1,10,2) , 'max_features':np.arange(0.1, 1, 1), 'random_state':np.arange(1,100,5)}

	return rgr, context_str, params, context_img

def one_model_reg_render(form, data_file, Y, context, test_file, Y_pred, Y_pred_test, target):
	X_test=test_file.loc[:, test_file.columns!=target]

	X_train=data_file.loc[:, data_file.columns!=target]
	Y_train=Y	
	
	context['x_test_cols']=X_test.columns
	context['tot_test_cols']=range(len(X_train.columns)+1)
	context['y_pred_test']=Y_pred_test
	Y_test_pred=np.matrix(Y_pred_test).T
	test_val_set=np.concatenate([X_test, Y_test_pred], axis=1)
	test_zip=np.concatenate([X_test, Y_test_pred],axis=1)
	context['full_test_set']=np.array(test_zip)

	context['x_cols']=X_train.columns
	context['tot_cols']=range(len(X_train.columns)+2)
	context['y_pred']=Y_pred 
	context['y_train']=Y_train 
	zip_val=list(zip(Y_train, Y_pred))
	Y_pred=np.matrix(Y_pred).T
	Y_train=np.matrix(Y_train).T
	val_set=np.concatenate([X_train, Y_train], axis=1)
	test_zip=np.concatenate([val_set, Y_pred],axis=1)
	context['full_set']=np.array(test_zip)

	context['explained_variance_score']=met.explained_variance_score(Y_train, Y_pred)
	context['mean_absolute_error']=met.mean_absolute_error(Y_train, Y_pred)
	context['mean_squared_error']=met.mean_squared_error(Y_train, Y_pred)
	context['mean_squared_log_error']=met.mean_squared_log_error(Y_train, Y_pred)
	context['median_absolute_error']=met.median_absolute_error(Y_train, Y_pred)
	context['r2_score']=met.r2_score(Y_train, Y_pred)
#			print(zip_val)
	context['zip_json']=json.dumps(zip_val)
	context['form']=form
	# here you can add things like:
	return render_to_response("result_reg.html",context)

def two_model_reg_render(form, data_file, Y, context, Y_pred_1, Y_pred_2, target):
	X_test=data_file.loc[:, data_file.columns!=target]
	Y_test=Y
	context['x_cols_1']=X_test.columns
	context['tot_cols_1']=range(len(X_test.columns)+2)
	context['y_pred_1']=Y_pred_1 
	context['y_test_1']=Y_test 
	zip_val=list(zip(Y_test, Y_pred_1))
	Y_pred_1=np.matrix(Y_pred_1).T
	Y_test=np.matrix(Y_test).T
	val_set=np.concatenate([X_test, Y_test], axis=1)
	test_zip=np.concatenate([val_set, Y_pred_1],axis=1)
	context['full_set_1']=np.array(test_zip)
	context['explained_variance_score_1']=met.explained_variance_score(Y_test, Y_pred_1)
	context['mean_absolute_error_1']=met.mean_absolute_error(Y_test, Y_pred_1)
	context['mean_squared_error_1']=met.mean_squared_error(Y_test, Y_pred_1)
	context['mean_squared_log_error_1']=met.mean_squared_log_error(Y_test, Y_pred_1)
	context['median_absolute_error_1']=met.median_absolute_error(Y_test, Y_pred_1)
	context['r2_score_1']=met.r2_score(Y_test, Y_pred_1)
	context['zip_json_1']=json.dumps(zip_val)

	X_test=data_file.loc[:, data_file.columns!=target]
	Y_test=Y
	context['x_cols_2']=X_test.columns
	context['tot_cols_2']=range(len(X_test.columns)+2)
	context['y_pred_2']=Y_pred_2
	context['y_test_2']=Y_test 
	zip_val=list(zip(Y_test, Y_pred_2))
	Y_pred_2=np.matrix(Y_pred_2).T
	Y_test=np.matrix(Y_test).T
	val_set=np.concatenate([X_test, Y_test], axis=1)
	test_zip=np.concatenate([val_set, Y_pred_2],axis=1)
	context['full_set_2']=np.array(test_zip)
	context['explained_variance_score_2']=met.explained_variance_score(Y_test, Y_pred_2)
	context['mean_absolute_error_2']=met.mean_absolute_error(Y_test, Y_pred_2)
	context['mean_squared_error_2']=met.mean_squared_error(Y_test, Y_pred_2)
	context['mean_squared_log_error_2']=met.mean_squared_log_error(Y_test, Y_pred_2)
	context['median_absolute_error_2']=met.median_absolute_error(Y_test, Y_pred_2)
	context['r2_score_2']=met.r2_score(Y_test, Y_pred_2)
#			print(zip_val)
	context['zip_json_2']=json.dumps(zip_val)

	context['form']=form
	# here you can add things like:
	return render_to_response("result_reg_comp.html",context)

def mapping_clust(method_clust, n_clusters):
	clust=None
	name=""
	if method_clust == 'KM':
		clust=KMeans(n_clusters)
		name="K-Means clustering"
	if method_clust == 'SC':
		clust=SpectralClustering(n_clusters)
		name="Spectral Clustering"
	if method_clust == 'AC':
		clust=AgglomerativeClustering(n_clusters)
		name="Agglomerative Clustering"
	if method_clust == 'BC':
		clust=Birch(n_clusters)
		name="Birch clustering"
	return clust, name

def one_model_clust_render(form, data_file, X, clust, context, Y_pred):
	context['x_cols']=data_file.columns
	context['tot_cols']=range(len(data_file.columns)+1)
	context['y_pred']=clust.labels_
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

def two_model_clust_render(form, data_file, X, clust1, clust2, context, Y_pred_1, Y_pred_2):
	context['x_cols_1']=data_file.columns
	context['tot_cols_1']=range(len(data_file.columns)+1)
	context['y_pred_1']=clust1.labels_
	tot_zip_1=np.concatenate([data_file, Y_pred_1], axis=1)
	context['full_set_1']=np.array(tot_zip_1)
	context['silhouette_1']=met.silhouette_score(X, labels=Y_pred_1)
	context['chscore_1']=met.calinski_harabaz_score(X, labels=Y_pred_1)

	pca=PCA(n_components=2).fit(X)
	pca_2d=pca.transform(X)
	#print(X.shape)
	#print(pca_2d)
	y_pred_1=[]
	for r in clust1.labels_:
		y_pred_1.append("Cluster "+str(r))
	full_set_1=np.concatenate([pca_2d, pd.DataFrame(np.matrix(y_pred_1)).T], axis=1)
	#print(full_set)
	context['zip_json_1']=json.dumps(full_set_1.tolist())
	
	context['x_cols_2']=data_file.columns
	context['tot_cols_2']=range(len(data_file.columns)+1)
	context['y_pred_2']=clust2.labels_
	tot_zip_2=np.concatenate([data_file, Y_pred_2], axis=1)
	context['full_set_2']=np.array(tot_zip_2)
	context['silhouette_2']=met.silhouette_score(X, labels=Y_pred_2)
	context['chscore_2']=met.calinski_harabaz_score(X, labels=Y_pred_2)

	pca=PCA(n_components=2).fit(X)
	pca_2d=pca.transform(X)
	#print(X.shape)
	#print(pca_2d)
	y_pred_2=[]
	for r in clust2.labels_:
		y_pred_2.append("Cluster "+str(r))
	full_set_2=np.concatenate([pca_2d, pd.DataFrame(np.matrix(y_pred_2)).T], axis=1)
	#print(full_set)
	context['zip_json_2']=json.dumps(full_set_2.tolist())

	context['form']=form
	return render_to_response("result_clust_comp.html", context)

def process(data_file, test_file, context, form, **kwargs):

	X=data_file.fillna(0)
	X_test=test_file.fillna(0)
	print("In process")
	algorithm_choice=form.cleaned_data['algorithm_choice']
	if algorithm_choice == 'S':
		target=form.cleaned_data['target']
		X=data_file.loc[:, data_file.columns!=target]
		for i in X.columns:
			if isinstance(X.at[0,i],str):
				X, col=encode_target(X, i)
				X=X.loc[:, X.columns!=i]

		Y=data_file[str(target)]

		X_test = test_file.loc[:, test_file.columns!=target]
		for i in X_test.columns:
			if isinstance(X_test.at[0,i],str):
				X_test, col=encode_target(X_test, i)
				X_test=X_test.loc[:, X_test.columns!=i]

		validation_split=form.cleaned_data['validation_split']
		method_super=form.cleaned_data['method_super']

		if method_super == 'C':
			method_class=form.cleaned_data['method_class']
			if len(method_class)==1:
				clf, context['algo_name'], params, context['img_url']=mapping_class(method_class[0])
				is_hyper=form.cleaned_data['is_class_hyper']
				if is_hyper == 'Y':
					grid_search=RandomizedSearchCV(clf, params)
					Y_pred=cross_val_predict(grid_search, X, Y, cv=int(100-validation_split))
					grid_search.fit(X, Y)
					Y_pred_test=grid_search.predict(X_test)
					context['hyper_result']=pd.DataFrame(grid_search.cv_results_).to_html()
				else:
					clf.fit(X, Y)
					Y_pred_test=clf.predict(X_test)					
					Y_pred=cross_val_predict(clf, X, Y, cv=int(100-validation_split)) #clf_gini.predict(X_test)
				return one_model_class_render(form, data_file, Y, context, test_file, Y_pred, Y_pred_test, target)
			if len(method_class)==2:
				clf_1, context['algo_name_1'], params_1, context['img_url_1']=mapping_class(method_class[0])
				is_hyper=form.cleaned_data['is_class_hyper']
				if is_hyper == 'Y':
					grid_search=RandomizedSearchCV(clf_1, params_1)
					Y_pred_1=cross_val_predict(grid_search, X, Y, cv=int(100-validation_split))
					grid_search.fit(X, Y)
					Y_pred_test_1=grid_search.predict(X_test)
					context['hyper_result_1']=pd.DataFrame(grid_search.cv_results_).to_html()
				else:
					clf_1.fit(X, Y)
					Y_pred_test_1=clf_1.predict(X_test)
					Y_pred_1=cross_val_predict(clf_1, X, Y, cv=int(100-validation_split))
				clf_2, context['algo_name_2'], params_2, context['img_url_2']=mapping_class(method_class[1])
				is_hyper=form.cleaned_data['is_class_hyper']
				if is_hyper == 'Y':
					grid_search=RandomizedSearchCV(clf_2, params_2)
					Y_pred_2=cross_val_predict(grid_search, X, Y, cv=int(100-validation_split))
					grid_search.fit(X, Y)
					Y_pred_test_2=grid_search.predict(X_test)
					context['hyper_result_2']=pd.DataFrame(grid_search.cv_results_).to_html()
				else:
					clf_2.fit(X, Y)
					Y_pred_test_2=clf_2.predict(X_test)					
					Y_pred_2=cross_val_predict(clf_2, X, Y, cv=int(100-validation_split))
				return two_model_class_render(form, data_file, Y, context, test_file, Y_pred_1, Y_pred_test_1, Y_pred_2, Y_pred_test_2, target)


		if method_super == 'R':
			method_reg=form.cleaned_data['method_reg']
			if len(method_reg) == 1:
				rgr, context['algo_name'], params, context['img_url']=mapping_reg(method_reg[0])
				print(method_reg)				
				print(rgr)
				is_hyper=form.cleaned_data['is_reg_hyper']
				if is_hyper == 'Y':
					grid_search=RandomizedSearchCV(rgr, params)
					Y_pred=cross_val_predict(grid_search, X, Y, cv=int(100-validation_split))
					grid_search.fit(X, Y)
					Y_pred_test=grid_search.predict(X_test)
					context['hyper_result']=pd.DataFrame(grid_search.cv_results_).to_html()
				else:			
					rgr.fit(X, Y)
					Y_pred_test=rgr.predict(X_test)					
					Y_pred=cross_val_predict(rgr, X, Y, cv=int(100-validation_split))
				return one_model_reg_render(form, data_file, Y, context, test_file, Y_pred, Y_pred_test, target)
			if len(method_reg) == 2:
				rgr_1, context['algo_name_1'], params_1, context['img_url_1']=mapping_reg(method_reg[0])
				is_hyper=form.cleaned_data['is_reg_hyper']
				if is_hyper == 'Y':
					grid_search=RandomizedSearchCV(rgr_1, params_1)
					grid_search.fit(X, Y)
					Y_pred_1=grid_search.predict(X)
					context['hyper_result_1']=pd.DataFrame(grid_search.cv_results_).to_html()
				else:			
					Y_pred_1=cross_val_predict(rgr_1, X, Y, cv=int(100-validation_split))
				rgr_2, context['algo_name_2'], params_2, context['img_url_2']=mapping_reg(method_reg[1])
				is_hyper=form.cleaned_data['is_reg_hyper']
				if is_hyper == 'Y':
					grid_search=RandomizedSearchCV(rgr_2, params_2)
					#Y_pred=cross_val_predict(grid_search, X, Y, cv=int(100-validation_split))
					grid_search.fit(X, Y)
					Y_pred_2=grid_search.predict(X)
					context['hyper_result_2']=pd.DataFrame(grid_search.cv_results_).to_html()
				else:			
					Y_pred_2=cross_val_predict(rgr_2, X, Y, cv=int(100-validation_split))
				return two_model_reg_render(form, data_file, Y, context, Y_pred_1, Y_pred_2, target)

	if algorithm_choice == 'U':
		method_unsuper=form.cleaned_data['method_unsuper']
		for i in X.columns:
			if isinstance(X.at[0,i],str):
				X, col=encode_target(X, i)
				X=X.loc[:, X.columns!=i]
		if method_unsuper=='C':
			method_clust=form.cleaned_data['method_clust']
			n_clusters=form.cleaned_data['no_clusters']
			if len(method_clust) == 1:
				cl, context["algo_name"]=mapping_clust(method_clust[0], n_clusters)
				clust=cl.fit(X)
				Y_pred=np.matrix(clust.labels_).T
				return one_model_clust_render(form, data_file, X, clust, context, Y_pred)
			if len(method_clust) == 2:
				cl1, context["algo_name_1"] = mapping_clust(method_clust[0], n_clusters)
				clust1=cl1.fit(X)
				Y_pred_1=np.matrix(clust1.labels_).T
				cl2, context["algo_name_2"] = mapping_clust(method_clust[1], n_clusters)
				clust2=cl2.fit(X)
				Y_pred_2=np.matrix(clust2.labels_).T
				return two_model_clust_render(form, data_file, X, clust1, clust2, context, Y_pred_1, Y_pred_2)
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
		testfile=form.cleaned_data['test_file']
		test_file=pd.DataFrame(pd.read_csv(testfile, sep=',', keep_default_na=False))
		temp=process(data_file, test_file, context, form, **kwargs)
		self.object=form.save()
		return temp

	def post(self, request, *args, **kwargs):
		self.object=None
		form_class = self.get_form_class()
		form = self.get_form(form_class)
		print('Form post')
		if form.is_valid():
			print('Form valid')
			return self.form_valid(form, **kwargs)
		else:
			print('Form invalid')
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
			'test_file': forms.FileInput,	
			'algorithm_choice': forms.Select(attrs={'onchange':"showDiv(this);"}),
			'method_super': forms.Select(attrs={'onchange':"showSuperDiv(this);"}),
			'method_unsuper': forms.Select(attrs={'onchange':"showUnsuperDiv(this);"}),
			'is_class_ensemble': forms.Select(attrs={'onchange':"showEnsembleClassDiv(this);"}),
			'is_reg_ensemble': forms.Select(attrs={'onchange':"showEnsembleRegDiv(this);"}),
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
				if method_super == 'C':
					method_cl=cleaned_data.get("method_class")
					if method_cl is None:
						raise ValidationError(('Need to select atleast 1 model'))
					if len(method_cl) > 2:
						raise ValidationError(('Can\'t select more than 2 models.'))				
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
		testfile=form.cleaned_data['test_file']
		test_file=pd.DataFrame(pd.read_csv(testfile, sep=',', keep_default_na=False))
		#print(form.cleaned_data["no_features"])
		temp=process(data_file, test_file, context, form, **kwargs)
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
