# Backward Elimination (using MLxtend) & Forward Elimination (using MLxtend)

import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector

dataset=pd.read_csv("C:\\Users\\ojhas\\Dropbox\\PC\\Downloads\\diabetes.csv")
# print(dataset)

x=dataset.iloc[:,:-1]
y=dataset["outcome"]

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

fs=SequentialFeatureSelector(lr,k_features=5,forward=True)  # Forward Elimination
# fs=SequentialFeatureSelector(lr,k_features=5,forward=False)  # Backward Elimination


fs.fit(x,y)


print(fs.feature_names)
print()
print(fs.k_feature_names_)
print()
print(fs.k_score_)