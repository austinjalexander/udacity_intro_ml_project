# _Udacity_ Intro ML Final Project

## Answers to Project Questions

1) All missing values (encoded as 'NaN's) were imputed with their respective columnar medians. Outliers were present in the data. By reviewing those outliers with columnar values in the 99th percentile, it became clear that one of the records did not represent an individual but the calculated totals of each column. This _total_ row was removed from the data set, and it was the only row removed.

2) I attempted to create a new feature. For example, the new feature based on Katie's suggestion (i.e., the percentage of emails sent to POIs relative to the total number of sent emails). Unfortunately, this feature and all others I tried failed to produce better performance results than without using new features at all. For instance, using **GaussianNB** as a control classifier (and without performing PCA), when adding the feature suggested by Katie, _Udacity's_ tester returned a precision score of 0.44612 and a recall score of 0.28150, the same scores without the additional feature. Thus, without a better mechanism for creating features other than randomly multyplying and dividing columns together (which is what I tried numerous times), I decided to move on without a new feature.

**SelectKBest** was used to rank the apparent importance of features and remove apparently superfluous ones. The value of k was tested  at 2, 4, 6, 8, and 10 using **GaussianNB** as a control classifier (and without performing PCA). The precision and recall values, suprisingly, were the same for each k value: precision: 0.44612, recall: 0.28150.

For preprocessing, all features were scaled using **StandardScaler**. After scaling, **RandomizedPCA** was used to perform dimensionality reduction and orthogonal transformation on the remaining features. By adding **RandomizedPCA**, allowing **GridSearchCV** to optimize its parameters (see below), using **GaussianNB** as a control classifier, the resulting precision and recall values were 0.50772 and 0.27950, respectively.

3) The classifiers attempted were: **GaussianNB** and **DecisionTreeClassifier**.

4) Most algorithms have parameters that should be tuned in order to modify their training performance, which is in part dependent on the nature of the data set under consideration.

For both preprocessing and classification, **GridSearchCV** was implemented to tune various parameters for each preprocessor and classifier (where applicable) as **GridSearchCV** attempts to find the best parameters. As a few examples of the parameters tuned (as there were many), the number of components for **RandomizedPCA**, the _k_ value for **SelectKBest**, the number of features to consider at split locations for **DecisionTreeClassifier**, and the penalty parameter for **SVC**.

5) Breaking the data into training and testing sets is a component of a statistical assessment method called _cross-validation_, which is an effort to estimate how well a model will perform/generalize when it encounters _new_ data. To this end in the current context, data sets used for training machine-learning models are broken up into separate _training_ and _testing_ sets. After models are trained with the training data alone, models are then evaluated using only the testing data.

As the current data set under consideration includes only 146 records (only 18 of which are identified as POIs), **StratifiedShuffleSplit** was used via **GridSearchCV** during the initial preprocessor and classifier evaluation phase since it allows for the possiblity of training and testing on all of the data available (i.e., as opposed to training only on a single training set and testing only on a single testing set), averaging the test results of each learning session in order to offer a clearer picture of model performance.

An additional note regarding the importance of techniques like **StratifiedShuffleSplit** and the way in which they have been implemented: if the data separation process described above is not randomized, it is possible to partition the data in such a way that the training and testing sets do not both properly represent the data in a balanced manner (e.g., when experimenting on the Enron data set, it would be possible to create a training set containing only non-POI-related data and a testing set with only POI-related data). 

6) Given the distribution of the final testing set, in particular the small number of POIs, and likely in part due to the binary nature of this specific classification problem, the generic score (i.e., the quotient of the number of correct predictions and the total number of predictions) for each classifier is easily made higher when no POIs are identified at all (i.e., all predicted labels are 0). 

Thus, in this case, using a _confusion matrix_ (and the resulting precision, recall, and F1 scores) is a better indicator of model/prediction performance. 

The highest-scoring combination of preprocessors and classifier was the following set: the **FeatureUnion** of **RandomizedPCA** (parameters: copy=True, iterated_power=1, n_components=6, random_state=42, whiten=True) and **SelectKBest** (k=8) and the classifier **GaussianNB** (no parameters).

This combination had a POI precision score (which is the quotient of the number of true positives and the sum of that number and the number of false positives) of 0.67 and a POI recall score (which is the quotient of the number of true positives and the sum of that number and the number of false negatives) of 0.40.

While unable to do so with the _Udacity_ tester, locally, in order to improve the score, the top-two highest-scoring combinations of preprocessors and classifiers were combined, which resulted in a POI precision score of 0.60 and a recall score of 0.60. This final process as well as all of the exploratory and original work for this project may be found at: https://github.com/austinjalexander/py/blob/master/nanodegree/intro_ml/EnronPOI-FinalProject.ipynb.

---

_I hereby confirm that this submission is my work. I have cited above the origins of any parts of the submission that were taken from Websites, books, forums, blog posts, github repositories, etc._

Aside from standard usage of scikit-learn and Python documentation/examples, none of the work above is dependent on any particular source and, thus, the work is my own.

---

## Epilogue: email text as data

I had hoped to use email-text data along with the financial data, but clearly there is a discrepancy between the individuals represented by the financial data and the email data. Thus, it would be difficult to join the two data sets in a meaningful way due to their lack of overlap. 

After this project, I may spend some time with text-vectorization of the email corpus and examine things like word frequencies; however, with such an apparent low number of known POI-emails being available (only 3!?), it's entirely unclear how useful such an endeavor would be for identifying POIs, although there are surely other interesting insights to be gleaned.
