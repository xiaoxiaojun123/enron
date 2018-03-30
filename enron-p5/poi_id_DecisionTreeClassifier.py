#!/usr/bin/python

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
financial_feature=['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                   'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                   'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
email_feature=['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages',
               'from_this_person_to_poi','shared_receipt_with_poi']
poi_feature=['poi']


features_list = ['poi','bonus','deferred_income','expenses'] # You will need to use more features
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
data=data_dict.pop("TOTAL",0)
data=data_dict.pop("TRAVEL AGENCY IN THE PARK",0)
data=data_dict.pop("LOCKHART EUGENE E",0)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Example starting point. Try investigating other evaluation techniques!
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn import tree
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedShuffleSplit

clf=tree.DecisionTreeClassifier()
parameters={'criterion':('gini','entropy'),
            'min_samples_split':[2,5,10]}

cv=StratifiedShuffleSplit(labels,50,random_state=42)
grid=GridSearchCV(clf,parameters,cv=cv,scoring='f1')

clf=grid.fit(features,labels)

print "best estimator:",grid.best_estimator_
print "best score:", grid.best_score_
clf=grid.best_estimator_



dump_classifier_and_data(clf, my_dataset, features_list)