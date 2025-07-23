# Polynomial Feature (Complettion of Previous One)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset=pd.read_csv("C:\\Users\\ojhas\\Dropbox\\PC\\Downloads\\mix.csv")

x=dataset.iloc[:,:-1]
y=dataset["output"]

from sklearn.preprocessing import PolynomialFeatures

pf=PolynomialFeatures(degree=2)
pf.fit(x)
x=pd.DataFrame(pf.transform(x))

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=10)

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()
lr.fit(x_train,y_train)

print(lr.score(x_test,y_test)*100)





