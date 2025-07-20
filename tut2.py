# Regression Analysis - Linear Regression Algorithm (Simple Linear)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# dataset=pd.read_csv("C:\\Users\\ojhas\\Dropbox\\PC\\Downloads\\package.csv")

data = {
    'cgpa': [5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8,
             7.0, 7.2, 7.4, 7.6, 7.8, 8.0, 8.2, 8.4, 8.6, 8.8,
             9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 10.2, 10.4, 10.6, 10.8],
    
    'ctc': [9.3, 10.5, 9.7, 11.4, 11.0, 12.5, 11.9, 12.1, 13.6, 13.6,
            13.9, 13.9, 14.5, 14.6, 14.6, 14.7, 15.9, 15.0, 16.4, 15.3,
            16.7, 17.8, 17.0, 18.6, 17.3, 18.5, 18.6, 19.4, 19.1, 19.6]
}
dataset = pd.DataFrame(data)

print(dataset.isnull().sum())

x=dataset[["cgpa"]]
y=dataset["ctc"]

# Train Test Split in Data Set

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.20,random_state=42)

# Simple Linear Regression

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(x_train,y_train)


# Y = m*X + C (Simple Linear Regression Equation)

print(lr.coef_)  # Value of m (slope)
print()
print(lr.intercept_) # Value of C (intercept)

# Y = 1.66494312*x + 1.684276973457429

print(lr.score(x_test,y_test)*100)

print(lr.predict([[5.2]]))
print()
print(1.66494312*5.2 + 1.684276973457429)  # Check the values 

y_pred=lr.predict(x)

sns.scatterplot(x="cgpa",y="ctc",data=dataset)
plt.plot(dataset["cgpa"],y_pred,c="red")
plt.legend(["org data","predict line"])
plt.show()


