# Logistic Regression (practical) (Multiclass Classification)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset=pd.read_csv("C:\\Users\\ojhas\\Dropbox\\PC\\Downloads\\plant_data.csv")
# print(dataset)

x=dataset.iloc[:,:-1]
y=dataset["species"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=10)


from sklearn.linear_model import LogisticRegression

# OVR method

lr=LogisticRegression(multi_class="ovr")
lr.fit(x_train,y_train)
print(lr.score(x_test,y_test)*100)

print()

# MULTINOMIAL method

lr=LogisticRegression(multi_class="multinomial")
lr.fit(x_train,y_train)
print(lr.score(x_test,y_test)*100)

print()

# Direct method

lr=LogisticRegression()
lr.fit(x_train,y_train)
print(lr.score(x_test,y_test)*100)

sns.pairplot(data=dataset,hue="species")
plt.show()

