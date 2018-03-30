#!/usr/bin/python

import pandas as pd
import matplotlib.pyplot
import sys
import pickle

sys.path.append("../tools/")

from sklearn import metrics
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'to_messages', 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# Example starting point. Try investigating other evaluation techniques!
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels,
                                                                                             test_size=0.3,
                                                                                             random_state=42)

clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
importances = clf.feature_importances_
print importances

key_index = 0
for index, feature in enumerate(importances):
    if feature > 0.1:
        key_index = index
        print index, feature
