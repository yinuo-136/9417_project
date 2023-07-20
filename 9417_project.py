import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
from sklearn.preprocessing import StandardScaler  
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split 

train_set = pd.read_csv('dataset/train.csv')

# Features engineering

# 1 filling up all empty cells

print(train_set.columns[train_set.isnull().sum() > 0])
# Output: Index(['BQ', 'CB', 'CC', 'DU', 'EL', 'FC', 'FL', 'FS', 'GL'], dtype='object')

# To minimize the effect of the filling data, we've deceided to fill in all empty cells by the mean of the column

columns_has_empty = ['BQ', 'CB', 'CC', 'DU', 'EL', 'FC', 'FL', 'FS', 'GL']

train_set[columns_has_empty] = train_set[columns_has_empty].fillna(train_set[columns_has_empty].mean())


# 2 parameters filtering
# The idea of this phase is to remove the features that have too high or too low correlations with the target values

# make the copy of the train_set, remove columns that contain strings
train_set_copy = train_set.copy()
train_set_copy.drop(['EJ', 'Id'], axis=1, inplace=True)
correlation_matrix = train_set_copy.corr()

class_correlations = correlation_matrix['Class']

low_correlation_params = class_correlations[abs(class_correlations) < 0.04]

high_correlation_params = class_correlations[abs(class_correlations) > 0.9]

print("Columns with low correlation (absolute value < 0.04):")
for column, correlation in low_correlation_params.items():
    print(f"{column}: {abs(correlation)}")

# output: 
#   AZ: 0.013515606981951173
#   CB: 0.014772394018243899
#   CH: 0.008144308808507026
#   CL: 0.016852143236700493
#   DN: 0.008477749950987764
#   DV: 0.015477478340990326
#   EG: 0.024609863760996505
#   EU: 0.03973900264478658
#   FC: 0.030571410396380175
#   FS: 0.0011340450175046388
#   GH: 0.03353967779212312

print("\nColumns with high correlation (absolute value > 0.8):")
for column, correlation in high_correlation_params.items():
    print(f"{column}: {abs(correlation)}")

# output:
#  Class: 1.0

# Since there is no params that has too high correlation, we will just remove all low correlation columns
col_drops = ['Id','AH', 'AZ','CB', 'CH','CL','CS','DN','DV','EG','EU','FC','FS','GH'] # also drop the id for future training reason
train_set.drop(col_drops,axis=1,inplace=True) 


# 3 Label encoding for EJ
# Since EJ is not a numerical data column, therefore we need a way to convert it to numerical state for training purpose
# we choose to use label encoding here
label = LabelEncoder()
target_cols = ['EJ'] 
train_set[target_cols] = train_set[target_cols].apply(label.fit_transform)


# 4 standardization
# Since the features in this dataset is not a small number, we consider to use stanardization to 
# scale the dataset to ensure each feature contributes relatively equal to the analysis. Also 
# reduce the effect of outliers
numeric_cols = [data_type for _,data_type in enumerate(train_set.select_dtypes(include=np.number).columns.tolist()) if(data_type!='Class')]
sc = StandardScaler()
train_set[numeric_cols] = sc.fit_transform(train_set[numeric_cols])


#5 split dataset into training and testing
# Becuase we do not have the test dataset at the moment, therefore we've decided to split the provided training dataset into 
# a new training set and a testing set

# split the last "Class" feature into Y, others(params) go to X
X = train_set.iloc[:, :-1].values 
y = train_set.iloc[:, -1].values

# our training and testing ratio will be 8:2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

