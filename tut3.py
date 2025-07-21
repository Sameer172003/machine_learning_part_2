# Multiple Linear Regression

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset=pd.read_csv("C:\\Users\\ojhas\\Dropbox\\PC\\Downloads\\multiple.csv")

# print(dataset)
x=dataset.iloc[:,:-1]
y=dataset["Salary"]

# Train Test Split in Data Set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.20,random_state=42)

# Multiple Linear Regression

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(x_train,y_train)

# Y = m1*X1 + m2*X2 + C (Multiple Linear Regression Equation)

print(lr.coef_)  # Value of m (slope)
print()
print(lr.intercept_) # Value of C (intercept)

# Y = -152.05476802*Age + 702.52417243*Experience + 37200.3001118789


print(lr.score(x_test,y_test)*100)
print()
print(-152.05476802*29 + 702.52417243*64 + 37200.3001118789)
print()
print(lr.predict([[29,64]]))

y_pred=lr.predict(x)

sns.pairplot(data=dataset)

plt.show()