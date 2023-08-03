import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split 
import matplotlib as plt

data = pd.read_csv('dataset/train.csv')

# # # # # # # # # # # # # # # #
# # # # # Preprocessing # # # #
# # # # # # # # # # # # # # # #

## stage 1 ##
# filling up missing data
print(data.columns[data.isnull().sum() > 0])

# to minimize the effect of the filling data,
# we've deceided to fill in all empty cells by the mean of the column
columns_has_empty = ['BQ', 'CB', 'CC', 'DU', 'EL', 'FC', 'FL', 'FS', 'GL']
data[columns_has_empty] = data[columns_has_empty].fillna(data[columns_has_empty].mean())

# validate the completeness of filling
print(data.columns[data.isnull().sum() > 0])

## stage 2 ##
# hot encoding the binary column 'EJ' by (A = 0, B = 1)
data.loc[data['EJ'] == 'A', 'EJ'] = 1
data.loc[data['EJ'] == 'B', 'EJ'] = 0

features_names = ['AB', 'AF', 'AH', 'AM', 'AR', 'AX', 'AY', 'AZ', 'BC', 'BD ', 'BN',
       'BP', 'BQ', 'BR', 'BZ', 'CB', 'CC', 'CD ', 'CF', 'CH', 'CL', 'CR', 'CS',
       'CU', 'CW ', 'DA', 'DE', 'DF', 'DH', 'DI', 'DL', 'DN', 'DU', 'DV', 'DY',
       'EB', 'EE', 'EG', 'EH', 'EJ', 'EL', 'EP', 'EU', 'FC', 'FD ', 'FE', 'FI',
       'FL', 'FR', 'FS', 'GB', 'GE', 'GF', 'GH', 'GI', 'GL']
target_name = 'Class'

# %%
## stage 3 ##
# dealing with outliers
# build a zero matrix used to record outliers
matrix = np.zeros((data.shape[0], data.shape[1]), dtype = object)
matrix[:, 0] = data['Id'].tolist()

def find_outliers(data):
    # Step 1: Calculate Q1, Q3, and IQR
    # Sort the data based on the second column (index 1)
    sorted_data = data[data[:, 1].argsort()]

    q1 = np.percentile(sorted_data[:, 1], 25)
    q3 = np.percentile(sorted_data[:, 1], 75)
    iqr = q3 - q1

    # Step 2: Find lower and upper bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Step 3: Identify outliers in the dataset
    outliers = sorted_data[(sorted_data[:, 1] < lower_bound) | (sorted_data[:, 1] > upper_bound)]
    
    return outliers

# record all outliers as 1 in zero matrix
i = 0
for index, features in enumerate(data.columns.tolist()):
    if features  != 'Id' and features  != 'EJ' and features  != 'Class':
        outliers = find_outliers(data.iloc[:, [0, i]].values)
        
        j = 0
        for id in matrix[:, 0]:
            if id in outliers[:, 0]:
                matrix[j, i] = 1
            j += 1

    i += 1

# compute the number of features are outlier for each observation
num_outliers = []
for entry in matrix:
    entry_sum = np.sum(entry[1:])
    num_outliers.append(entry_sum)
    
matrix = np.column_stack((matrix, num_outliers))

# find these observations have too many outlier features and remove them
# Train = pd.concat([data, pd.DataFrame(data, columns = ['Class'])], axis = 1)

outliers = find_outliers(matrix[:, [0, -1]])
outliers = pd.DataFrame(outliers[:, 0], columns=['Id'])
data = data[~data['Id'].isin(outliers.iloc[:, 0])]

X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

# %%
## stage 4 ##
# standardization train data
numeric_cols = [features for index, features in enumerate(X.columns.tolist()) if(features != 'Class' and features != 'EJ')]
sc = StandardScaler()
X[numeric_cols] = sc.fit_transform(X[numeric_cols])




# %%
# address the issue of imbalanced classes
from sklearn.metrics import log_loss
from sklearn.utils import class_weight

def balanced_log_loss(y_true, y_pred):
    class_weights = class_weight.compute_class_weight('balanced', classes = np.unique(y_true), y = y_true)
    weights = class_weights[y_true.astype(int)]
    loss = log_loss(y_true, y_pred, sample_weight = weights)
    return loss

# %%
def adjusted_prob(y_hat):
    i = 0
    for y_hat_classA, y_hat_classB in y_hat:

        y_hat_classA = np.max([np.min([y_hat_classA, 1 - 10 ** -15]), 10 ** -15])
        y_hat_classB = np.max([np.min([y_hat_classB, 1 - 10 ** -15]), 10 ** -15])

        y_hat[i, 0] = y_hat_classA
        y_hat[i, 1] = y_hat_classB
        i += 1
    
    return y_hat

# %%
# # # # # # # # # # # # # # # # # # # #
# # # # K-Fold Cross Validation # # # #
# # # # # # # # # # # # # # # # # # # #
from sklearn.model_selection import StratifiedKFold

n_splits = 10
cv_score_LR = 0
best_score = 100
best_model = 10
sen = 0
spe = 0
best_sen = 0
best_spe = 0
best_y_pred = 0

skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 42)

for k in range(1, 30):
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        knn_model = KNeighborsClassifier(n_neighbors = k, metric = 'minkowski')
        knn_model.fit(X_train, y_train)

        y_hat_test_LR = knn_model.predict_proba(X_test)
        y_hat_test_LR = adjusted_prob(y_hat_test_LR)
        cv_score_LR += balanced_log_loss(y_test, y_hat_test_LR)

        from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
        # Calculate accuracy
        y_pred = knn_model.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred)
        # Extract the values from the confusion matrix
        true_negatives = conf_matrix[0, 0]
        false_positives = conf_matrix[0, 1]
        false_negatives = conf_matrix[1, 0]
        true_positives = conf_matrix[1, 1]
        # Calculate sensitivity and specificity
        sensitivity = true_positives / (true_positives + false_negatives)
        specificity = true_negatives / (true_negatives + false_positives)
        sen += sensitivity
        spe += specificity
    
    if best_score > cv_score_LR:
        best_score = cv_score_LR
        best_model = knn_model
        best_y_pred = y_pred
    
    if best_sen < sen:
        best_sen = sen
    
    if best_spe < spe:
        best_spe = spe
    
    cv_score_LR = 0
    sen = 0
    spe = 0


print(f"avg_Sensitivity: {best_sen / 10}, avg_Specificity: {best_spe / 10}")


print('best CV score: ', best_score / 10)
print("best accuracy: ", np.mean(cross_val_score(best_model, X, y, cv=5))*100)
