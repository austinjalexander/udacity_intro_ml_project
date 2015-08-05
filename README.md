# udacity_intro_ml_project

## Answers to Project Questions

1) The goal of this project was to attempt to classify _persons of interest_ (POIs) using financial data and email-frequency counts collected beforehand by Katie (and perhaps other staff) at _Udacity_. There were outliers in the data (calculated as values below 0.5% and above 99.5%), and their removal prior to training classifiers changed the vertical shape of the data by about 7% (i.e., 10 examples were removed along with their corresponding labels; none of these removals was labeled as a POI). After removing outliers, all missing values (encoded as 'NaN's) were imputed with their respective columnar medians (which, based on multiple experiments, ended up being a better option for performance in this case over columnar means). 

2) I started out using all of the features provided. I experimented with a number of hand-engineered features based both on _Udacity_'s lectures and my own intuition; for example, among others, I attempted to use the quotient of _exercised_stock_options_ and _total_stock_value_, the quotient of _from_poi_to_this_person_ and _from_this_person_to_poi_, and the product of _loan_advances_ and _restricted_stock_ (which were two of the highest-scoring, non-email-related features after the cutoff point [$k=2$] from an initial implementation of **SelectKBest**). Out of all of the hand-engineered features, the product of _loan_advances_ and _restricted_stock_ had the greatest impact (evinced by its later **SelectKBest** score and the predictive performance provided to **DecisionTreeClassifier**). Thus, for the sake of computation and training performance, the other created features were discarded.

In addition to modifications noted above, I implemented a **Pipeline** wherein I first scaled all features using **MinMaxScaler** (due to the apparent mathematical importance of scaling), followed by either **RandomizedPCA** (to perform dimensionality reduction and orthogonal transformation on the available features) or **SelectKBest** (to omit any apparently superfluous features).

In a sense, I also auto-engineered the available features via **FeatureUnion** (combining **RandomizedPCA** and **SelectKBest**), but I was unable to see a performance benefit, while simultaneously seeing a signifcant increase in computation time.

Preprocessing via StandardScaler and the feature union of PCA (keeping 6 components) and SelectKBest ($k = 8$) followed by training a GaussianNB classifier provided the highest F1 score of 0.84.

Only 146 records; 18 are identified as POIs. Since 146 is such as small amound, stratified-shuffle split using gridsearch.

the created feature scored the highest; and without it, only score 

gridsearch tried pca, selectkbest, varrying list each

type out the values for each classifier

use different words to describe precision and recall

general discussion of cross-validation
___


----------------







The report says, "Cross-validation allows for the possiblity of training and testing on all of the data available (i.e., as opposed to training only on a single training set and testing only on a single testing set), averaging the test results of each learning session to offer a clearer picture of model performance"

This is really what StratifiedShuffleSplit or KFolds cross validation does. But that isn't the only way to validate a machine learning algorithm. If we had a very large, randomized and balanced data set, then doing only one train/test split would probably be an acceptable strategy.

Please give a more general discussion of what cross validation is. Splitting the data into a training and test set is part of cross validation. But it's also about making sure the algorithm performs well on data that it hasn't seen before.


