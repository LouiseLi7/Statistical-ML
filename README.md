# README
# Packages and Libraries:
# Read files:
import pandas as pd

import numpy as np

import io

from matplotlib import pyplot as plt


# Data cleaning packages:
from imblearn.over_sampling import SMOTE

from sklearn.impute import KNNImputer

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder


# Metrics evaluation packages:
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn import metrics

from sklearn.model_selection import GridSearchCV


# Algorithm packages:
## XGboost
from xgboost import XGBClassifier

from xgboost import plot_importance

## Adaboost
from sklearn.ensemble import AdaBoostClassifier

## RandomForest
from sklearn.ensemble import RandomForestRegressor

## Naive Bayes
from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import ComplementNB

## SVM
from sklearn import svm

## Neural Network
!pip install scikeras

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense

from keras.utils import to_categorical

from keras.optimizers import SGD

from scikeras.wrappers import KerasClassifier

# Logistic Regression
from sklearn.linear_model import LogisticRegression

```
I have already provided the .ipynb and .py file for the codes, you can run it.
I write it on colab so using colab should not have problem.
I am using text cells that will be easier to separate each part and scroll down. 

XGboost may take a long time to tune the parameters, and it may have different results due to the forest. 
Libraries without classifications: 
```
```
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn import svm
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

!pip install scikeras

from xgboost import XGBClassifier
from xgboost import plot_importance

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.optimizers import SGD
from scikeras.wrappers import KerasClassifier

from imblearn.over_sampling import SMOTE

from google.colab import files
uploaded = files.upload()

import io
data = pd.read_csv(io.BytesIO(uploaded["FINAL Animal Data 2022.csv"]))
```

