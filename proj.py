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
from sklearn.decomposition import RandomizedPCA
from sklearn.feature_selection import SelectKBest
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


