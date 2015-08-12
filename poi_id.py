# GENERAL IMPORTS AND MODS
import sys
import os
from time import time
import re
import pickle
sys.path.append("ud120-projects/tools/")
sys.path.append("ud120-projects/final_project/")
import numpy as np
import pandas as pd

# sklearn imports
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

from sklearn.cross_validation import StratifiedShuffleSplit

from sklearn.pipeline import make_pipeline

from sklearn.feature_selection import SelectKBest

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import RandomizedPCA

from sklearn.grid_search import GridSearchCV

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

### Load the dictionary containing the dataset
data_dict = pickle.load(open("ud120-projects/final_project/final_project_dataset.pkl", "r") )

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'bonus', 'deferred_income', 'exercised_stock_options', 'salary', 'total_stock_value']


### Task 2: Remove outliers
# remove apparent non-individual records from data set
del data_dict['THE TRAVEL AGENCY IN THE PARK']
del data_dict['TOTAL']

# impute 'NaN' values to columnar means
for feature in features_list[1:]:
    feature_values = []
    for record in data_dict.values():
        if record[feature] != 'NaN':
            feature_values.append(record[feature])
    
    median = np.median(feature_values)
    
    for key in data_dict.keys():
        if data_dict[key][feature] == 'NaN':
            data_dict[key][feature] = median

### Task 3: Create new feature(s)
new_feature = 'bonus_salary_ratio'
features_list.append(new_feature)
for key in data_dict.keys():
    data_dict[key][new_feature] = np.true_divide(data_dict[key]['bonus'],data_dict[key]['salary'])

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.
#clf = DecisionTreeClassifier()
#clf = SVC()

pipeline = make_pipeline(StandardScaler(), RandomizedPCA(), clf)

# cross validation    
cv = StratifiedShuffleSplit(labels, test_size=0.2, random_state=42)

# tune parameters
params = dict()

# for PCA
params['randomizedpca__iterated_power'] = [1, 2, 3]
params['randomizedpca__n_components'] = [2, 4, 6, 8, 10]
params['randomizedpca__random_state'] = [42]
params['randomizedpca__whiten'] = [True, False]

if str(clf)[0] == 'D':
    params['decisiontreeclassifier__criterion'] = ['gini', 'entropy']
    params['decisiontreeclassifier__max_features'] = ['auto', 'sqrt', 'log2', None]
    params['decisiontreeclassifier__class_weight'] = ['auto', None]
    params['decisiontreeclassifier__random_state'] = [42]
    
if str(clf)[0] == 'S':
    params['svc__C'] = [2**x for x in np.arange(-15, 15+1, 3)]
    params['svc__gamma'] = [2**x for x in np.arange(-15, 15+1, 3)]
    params['svc__random_state'] = [42]

grid_search = GridSearchCV(pipeline, param_grid=params, n_jobs=1, cv=cv)

grid_search.fit(features, labels)

clf = grid_search.best_estimator_

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)


