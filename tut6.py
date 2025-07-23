# Logistic Regression (practical) (Binary Classification)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset=pd.read_csv("C:\\Users\\ojhas\\Dropbox\\PC\\Downloads\\social.csv")

dataset.drop(columns="EstimatedSalary",inplace=True)
# print(dataset)

x=dataset[["Age"]]
y=dataset["Purchased"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()
lr.fit(x_train,y_train)

print(lr.score(x_test,y_test)*100)

print(lr.predict([[45]]))

sns.scatterplot(x="Age",y="Purchased",data=dataset)
sns.lineplot(x="Age",y=lr.predict(x),data=dataset,color="red")
plt.show()