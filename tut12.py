# Imbalanced dataset

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset=pd.read_csv("C:\\Users\\ojhas\\Dropbox\\PC\\Downloads\\dup.csv")
# print(dataset)
print(dataset["Purchased"].value_counts())

x=dataset.iloc[:,:-1]
y=dataset["Purchased"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()
lr.fit(x_train,y_train)

print(lr.score(x_test,y_test)*100)

# Under Sampling

from imblearn.under_sampling import RandomUnderSampler

ru=RandomUnderSampler()
ru_x,ru_y=ru.fit_resample(x,y)

print(ru_y.value_counts())


# Over Sampling

from imblearn.over_sampling import RandomOverSampler

ru=RandomOverSampler()
ru_x,ru_y=ru.fit_resample(x,y)

print(ru_y.value_counts())