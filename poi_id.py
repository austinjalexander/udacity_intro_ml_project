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

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import RandomizedPCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_union
from sklearn.pipeline import make_pipeline
from sklearn.grid_search import GridSearchCV

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.cluster import KMeans

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# LOAD DATA

### Load the dictionary containing the dataset
data_dict = pickle.load(open("ud120-projects/final_project/final_project_dataset.pkl", "r") )
my_dataset = data_dict

# FEATURE SELECTION
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

names = np.array(my_dataset.keys())
features_list = my_dataset.itervalues().next().keys()
features_list.sort()
features_list.remove('poi')
features_list.insert(0, 'poi')
features_list.remove('email_address')

# DATA-FORMAT CONVERSION
# convert dictionary to pandas dataframe
df = pd.DataFrame([entry for entry in my_dataset.itervalues()])
df = df.drop('email_address', axis=1)
df = df[features_list]
df.poi = df.poi.astype('int')
df = df.convert_objects(convert_numeric=True)

for col in list(df.columns):
    df[col] = df[col].round(decimals=3)

# SEPARATE LABELS FROM FEATURES
# create labels
y = df.poi.values

# create initial features
X = df.drop('poi', axis=1).values

# OUTLIER REMOVAL
### Task 2: Remove outliers
# hand-tuned to remove ~5% (in this case, 7%)
num_rows = X.shape[0]
num_cols = X.shape[1]
rows_to_remove = set()

for i in xrange(num_cols):
    point_five_percentile = np.percentile(X[:,i], 0.5)
    ninety_nine_point_five_percentile = np.percentile(X[:,i], 99.5)
    
    for j in xrange(num_rows):
        if X[j,i] < point_five_percentile or X[j,i] > ninety_nine_point_five_percentile:
            rows_to_remove.add(j)

X = np.delete(X, list(rows_to_remove), axis=0)
y = np.delete(y, list(rows_to_remove))
    
names = np.delete(names, list(rows_to_remove))

# 'NaN' IMPUTATION
# impute 'NaN' values to column means
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(X)
X = imp.transform(X)

imp_values = imp.statistics_

# FEATURE CREATION
### Task 3: Create new feature(s)

def create_new_feature(X, col1, col2, operation, feature_name):
    
    features_list.append(feature_name)
    
    new_col = []
    if operation == '*':
        new_col = (X[:,col1] * X[:,col2])
    elif operation == '/':
        new_col = np.true_divide(X[:,col1], X[:, col2])
    
    new_col.shape = (new_col.shape[0], 1)
    #print new_col.shape

    X = np.hstack((X, new_col))
    #print X.shape
    
    return X

X = create_new_feature(X, 0, 14, '*', 'selectkbest_product')

# MACHINE LEARN!

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

def grid_searcher(clf, pca_skb, output):
    
    t0 = time()
    
    even_range = range(2,X.shape[1],2)
    random_state = [42]
    t_or_f = [True, False]
    #powers_of_ten = [10**x for x in range(-5,5)]
    logspace = np.logspace(-5, 5, 10)
    criteria = ['gini', 'entropy']
    splitters = ['best', 'random']
    max_features = ['auto', 'sqrt', 'log2', None]
    
    # modify features, remove features via pipeline
    
    pipeline = []
    params = dict()
    pipeline_clf = ""
    
    if pca_skb == "pca_skb":
        pipeline = make_pipeline(StandardScaler(), make_union(RandomizedPCA(), SelectKBest()), clf)

        params = dict(featureunion__randomizedpca__n_components = even_range,
                      featureunion__randomizedpca__iterated_power = [1, 2, 3],
                      featureunion__randomizedpca__whiten = t_or_f,
                      featureunion__randomizedpca__random_state = random_state,
                      featureunion__selectkbest__k = even_range)   
        
    elif pca_skb == "pca":
        pipeline = make_pipeline(StandardScaler(), RandomizedPCA(), clf)

        params = dict(randomizedpca__n_components = [4],
                      randomizedpca__iterated_power = [1, 2, 3],
                      randomizedpca__whiten = t_or_f,
                      randomizedpca__random_state = random_state)   
        
    elif pca_skb == "skb":
        pipeline = make_pipeline(StandardScaler(), SelectKBest(), clf)

        params = dict(selectkbest__k = [4])   
    
    pipeline_clf = pipeline.steps[2][0]      
    
    if pipeline_clf == 'decisiontreeclassifier' or pipeline_clf == 'randomforestclassifier':
        params["{}__criterion".format(pipeline_clf)] = criteria
        #params["{}__splitter".format(pipeline_clf)] = splitters
        params["{}__max_features".format(pipeline_clf)] = max_features
        #params["{}__min_samples_split".format(pipeline_clf)] = even_range
        params["{}__class_weight".format(pipeline_clf)] = ['auto', None]
        params["{}__random_state".format(pipeline_clf)] = random_state
        
    # cross validation    
    cv = StratifiedShuffleSplit(y, test_size=0.2, random_state=random_state[0])
    
    # tune parameters
    grid_search = GridSearchCV(pipeline, param_grid=params, n_jobs=1, cv=cv)

    grid_search.fit(X, y)

    if output == True:
        print "*"*15, pipeline_clf.upper(), "*"*15
        print "\nBEST SCORE: ", grid_search.best_score_, "\n"
        print "\nBEST PARAMS: ", grid_search.best_params_, "\n"

    # split into training and testing data for reporting results
    if output == True:
        print "#"*50
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state[0])

    if output == True:
        print "\nBEST ESTIMATOR:"
    clf = grid_search.best_estimator_
    if output == True:
        print clf
    clf.fit(X_train, y_train)
    
    if pca_skb == "skb" or pca_skb == "pca_skb":
    
        if output == True:
            print "\nSelectKBest SCORES:"
        features = features_list[1:]
        
        selectkbest_scores = clf.steps[1][1].scores_ if pca_skb == "skb" else clf.steps[1][1].transformer_list[1][1].scores_
        
        selectkbest_scores = np.round(selectkbest_scores, 2)
        for i in xrange(len(features)):
            if output == True:
                print "\t", features[i], ": ", selectkbest_scores[i]
    
    if pipeline_clf == 'decisiontreeclassifier' or pipeline_clf == 'randomforestclassifier':
        if output == True:
            print "\n{} FEATURE IMPORTANCES:".format(pipeline_clf.upper())
            print clf.steps[2][1].feature_importances_
    
    if output == True:
        print "\n", "#"*50
    
        print "\nPREDICTIONS:"

        print "\nground truth:\n\t", y_test 
    
    y_pred = clf.predict(X_test)
    if output == True:
        print "\npredictions:\n\t", y_pred

        print "\nscore: ", clf.score(X_test, y_test)

        print "\nEVALUATIONS:"
        print "\nconfusion matrix:\n", confusion_matrix(y_test, y_pred)
    
        print "\nclassification report:\n", classification_report(y_test, y_pred, target_names=["non-poi", "poi"])
    
        print "ELAPSED TIME: ", round(time()-t0,3), "s"
    
    return clf


clf = grid_searcher(GaussianNB(), 'pca_skb', output=False)

#######################################################################
#######################################################################
#######################################################################

# PREPARE FOR UDACITY TESTER

# remove emails
for key in my_dataset.keys():
    my_dataset[key].pop('email_address')
    
# remove outliers from original data set
for key in my_dataset.keys():
    if key not in names:
        my_dataset.pop(key)

# replace 'NaN's
for key in my_dataset.keys():
    for sub_key in my_dataset[key].keys():
        if my_dataset[key][sub_key] == 'NaN':
            i = (df.columns.get_loc(sub_key) - 1)
            my_dataset[key][sub_key] = imp_values[i]
            
# add created feature
i = 0
for key in my_dataset.keys():
    my_dataset[key]['selectkbest_product'] = X[i,-1]
    i += 1

# use Udacity tester
print "\nUDACITY TESTER RESULTS: "
test_classifier(clf, my_dataset, features_list)

# UDACITY DUMP
dump_classifier_and_data(clf, my_dataset, features_list)

