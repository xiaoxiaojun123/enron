#!/usr/bin/python

# data explore
import matplotlib.pyplot
import pprint
import sys
import pickle

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    pprint.pprint(data_dict)


def get_size(data_dict):
    print "data_size:", len(data_dict)


def get_poi(data_dict):
    num_poi = 0
    for values in data_dict.values():
        if values['poi'] == True:
            num_poi += 1
    print "num_poi:", num_poi


def get_non_email(data_dict):
    num_email = 0
    num_salary = 0
    for values in data_dict.values():
        if values['email_address'] != 'NaN':
            num_email += 1
        if values['salary'] != 'NaN':
            num_salary += 1

    print "num_email:", num_email, "num_salary:", num_salary


get_size(data_dict)
get_poi(data_dict)
get_non_email(data_dict)


def count_non(data_dict):
    non = {}
    for person in data_dict:
        for key, value in data_dict[person].iteritems():
            if value == "NaN":
                if key in non:
                    non[key] += 1
                else:
                    non[key] = 1
    pprint.pprint(non)


count_non(data_dict)

### read in data dictionary, convert to numpy array
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter(salary, bonus)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

