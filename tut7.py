# Logistic Regression (practical) (Binary Classification) (Multiple input)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset=pd.read_csv("C:\\Users\\ojhas\\Dropbox\\PC\\Downloads\\placement.csv")
# print(dataset)

x=dataset.iloc[:,:-1]
y=dataset["placed"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=10)


from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()
lr.fit(x_train,y_train)

print(lr.score(x_test,y_test)*100)

print()

print(lr.predict([[8.5,5.09]]))

sns.scatterplot(x="cgpa",y="score",data=dataset,hue="placed")
plt.show()