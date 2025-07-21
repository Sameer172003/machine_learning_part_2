# Polynomial Regression

import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("C:\\Users\\ojhas\\Dropbox\\PC\\Downloads\\poly.csv")
# print(dataset)

x=dataset[["Level"]]
y=dataset["Salary"]

# Polynomial Regression 

from sklearn.preprocessing import PolynomialFeatures

pf=PolynomialFeatures(degree=2)

pf.fit(x)
x=pf.transform(x)

# Train Test Split in Data Set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.20,random_state=42)


# Linear Regression   

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(x_train,y_train)

# Y = m1*X1 + (m2*X2)^2 + C

print(lr.coef_)  # Value of m (slope)
print()
print(lr.intercept_) # Value of C (intercept)
print()

# Y = 977.54445272*X1 + (496.93729247*X2)^2 + 623.2626301502532

print(lr.score(x_test,y_test)*100)
print()


prd=lr.predict(x)

test=pf.transform([[5]])
print(x)
print()

print(lr.predict(test))
print()
print(977.54445272*5 + (496.93729247*25) + 623.2626301502532)



plt.scatter(dataset["Level"],dataset["Salary"])
plt.plot(dataset["Level"],prd,c="red")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.legend(["Original", "Prediction"])
plt.show()