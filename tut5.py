# L1 (Lasso Regularization), L2 (Ridge Regularization)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dataset=pd.read_csv("C:\\Users\\ojhas\\Dropbox\\PC\\Downloads\\housePrice.csv")

# print(dataset)

x=dataset.iloc[:,:-1]
y=dataset["price"]

# Scaling

sc=StandardScaler()
sc.fit(x)
x=pd.DataFrame(sc.transform(x),columns=x.columns)

# Train and Test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Regularization (L2 and L1)

from sklearn.linear_model import LinearRegression, Lasso, Ridge

from sklearn.metrics import mean_absolute_error, mean_squared_error

lr=LinearRegression()
lr.fit(x_train,y_train)
print(lr.score(x_test,y_test)*100)

print()

print(mean_squared_error(y_test,lr.predict(x_test)))
print(mean_absolute_error(y_test,lr.predict(x_test)))
print(np.sqrt(mean_squared_error(y_test,lr.predict(x_test))))

print()

# plt.figure(figsize=(15,5))
# plt.bar(x.columns,lr.coef_)
# plt.title("Linear Regression")
# plt.show()

# Lasso Regularization  

la=Lasso(alpha=0.0001)
la.fit(x_train,y_train)
print(la.score(x_test,y_test)*100)

print()

print(mean_squared_error(y_test,la.predict(x_test)))
print(mean_absolute_error(y_test,la.predict(x_test)))
print(np.sqrt(mean_squared_error(y_test,la.predict(x_test))))

print()

# plt.figure(figsize=(15,5))
# plt.bar(x.columns,la.coef_)
# plt.title("Lasso Regularization")
# plt.show()

# Ridge Regularization  

ri=Ridge(alpha=0.00000001)
ri.fit(x_train,y_train)
print(ri.score(x_test,y_test)*100)

print()

print(mean_squared_error(y_test,ri.predict(x_test)))
print(mean_absolute_error(y_test,ri.predict(x_test)))
print(np.sqrt(mean_squared_error(y_test,ri.predict(x_test))))

print()

# plt.figure(figsize=(15,5))
# plt.bar(x.columns,ri.coef_)
# plt.title("Ridge Regularization")
# plt.show()

df=pd.DataFrame({"Col_name":x.columns,"Linear Regression":lr.coef_,"Lasso":la.coef_,"Ridge":ri.coef_})
print(df)
