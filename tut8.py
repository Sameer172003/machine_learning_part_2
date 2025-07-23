# Logistic Regression (practical) (Binary Classification) (Polynomial input)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset=pd.read_csv("C:\\Users\\ojhas\\Dropbox\\PC\\Downloads\\mix.csv")
# print(dataset)

x=dataset.iloc[:,:-1]
y=dataset["output"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=10)


from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()
lr.fit(x_train,y_train)

print(lr.score(x_test,y_test)*100)

from mlxtend.plotting import plot_decision_regions

plot_decision_regions(x.to_numpy(),y.to_numpy(),clf=lr)
plt.show()

# sns.scatterplot(x="data1",y="data2",data=dataset,hue="output")
# plt.show()