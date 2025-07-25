# Naive Bayes

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

dataset=pd.read_csv("C:\\Users\\ojhas\\Dropbox\\PC\\Downloads\\dupli.csv")
# print(dataset)

x=dataset.iloc[:,:-1]
y=dataset["placed"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.20)

from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB

gnb=GaussianNB()
gnb.fit(x_train,y_train)
print(gnb.score(x_test,y_test)*100)
print()

bnb=BernoulliNB()
bnb.fit(x_train,y_train)
print(bnb.score(x_test,y_test)*100)
print()

mnb=MultinomialNB()
mnb.fit(x_train,y_train)
print(mnb.score(x_test,y_test)*100)
print()

# sns.kdeplot(data=dataset["cgpa"])
# plt.show()

# sns.scatterplot(x="cgpa",y="score",data=dataset,hue="placed")
# plt.show()
