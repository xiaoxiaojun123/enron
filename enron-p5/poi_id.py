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
financial_feature = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                     'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                     'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
email_feature = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi']
poi_feature = ['poi']

features_list = ['poi', 'salary', 'bonus', 'deferred_income', 'expenses', 'long_term_incentive',
                 'restricted_stock']  # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data = data_dict.pop("TOTAL", 0)
data=data_dict.pop("TRAVEL AGENCY IN THE PARK",0)
data=data_dict.pop("LOCKHART EUGENE E",0)

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

outlier = []
for key in data_dict:
    val = data_dict[key]["salary"]
    if val == "NaN":
        continue
    else:
        outlier.append((key, int(val)))

outlier_final = (sorted(outlier, key=lambda x: x[1], reverse=True)[:4])
### uncomment for printing top 4 salaries
### print outliers_final



### Task 3: Create new feature(s)
# new features are:fraction_from_poi_email,fraction_to_poi_email
def computeFraction(a, total):
    new_list = []
    for i in data_dict:
        if data_dict[i][a] == 'NaN' or data_dict[i][total] == 'NaN':
            new_list.append(0.)
        else:
            new_list.append(float(data_dict[i][a]) / float(data_dict[i][total]))
    return new_list


fraction_from_poi_email = computeFraction("from_poi_to_this_person", "to_messages")
fraction_to_poi_email = computeFraction("from_this_person_to_poi", "from_messages")

# insert new features into data_dict
import pprint

count = 0
for i in data_dict:
    data_dict[i]["fraction_from_poi_email"] = fraction_from_poi_email[count]
    data_dict[i]["fraction_to_poi_email"] = fraction_to_poi_email[count]
    count += 1
# pprint.pprint (data_dict)


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

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(features_train, labels_train)
y_pred = clf.predict(features_test)
recall = metrics.recall_score(labels_test, y_pred)
precision = metrics.precision_score(labels_test, y_pred)
print "The naive_bayes's recall is: %s " % recall
print "The naive_bayes's precision is : %s" % precision



dump_classifier_and_data(clf, my_dataset, features_list)