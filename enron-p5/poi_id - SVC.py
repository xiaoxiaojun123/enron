#!/usr/bin/python
import numpy as pf
import pandas as pd
import matplotlib.pyplot
import sys
import pickle

sys.path.append("../tools/")

from sklearn import metrics
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus', 'deferred_income', 'expenses', 'long_term_incentive',
                 'restricted_stock']  # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data = data_dict.pop("TOTAL", 0)
data=data_dict.pop("TRAVEL AGENCY IN THE PARK",0)
data=data_dict.pop("LOCKHART EUGENE E",0)

features_list = ['poi', 'salary', 'bonus', 'deferred_income', 'expenses', 'long_term_incentive',
                 'restricted_stock']  # You will need to use more features

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# Example starting point. Try investigating other evaluation techniques!


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifier
from sklearn import cross_validation
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

cv = StratifiedShuffleSplit(labels, 50,test_size=0.3, random_state=18)
steps = [('scale', MinMaxScaler()), ('SVC', SVC())]
clf = Pipeline(steps)

parameters = {'SVC__kernel': ['linear', 'rbf'],
              'SVC__C': [100,200,250,300],
              'SVC__gamma':[3,4,5,6],
              'SVC__decision_function_shape':['ovo'],
              'SVC__class_weight':['balanced']}
grid = GridSearchCV(clf, parameters, cv=cv, scoring='f1')
grid.fit(features, labels)

print "best estimator:", grid.best_estimator_
print "best score:", grid.best_score_
clf = grid.best_estimator_



dump_classifier_and_data(clf, my_dataset, features_list)