import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
sns.set()


data = pd.read_csv('resources/1.01.+Simple+linear+regression.csv')
print(data.head())

x = data['SAT']
y = data['GPA']

print(str(x.shape) + "\n" + str(y.shape))
x_matrix = x.values.reshape(-1, 1)
print(str(x_matrix.shape) + "\n" + str(y.shape))

reg = LinearRegression()
reg.fit(x_matrix, y)

# n_jobs is for multithreading if you have more CPU
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

# R-Squared
print(reg.score(x_matrix, y))
# Coeffience
print(reg.coef_)
# Intercept
print(reg.intercept_)
# Making predictions
#print(reg.predict(1740)) #this line of code does not work
# predicts GPA according to the SAT score

new_data = pd.DataFrame(data=[1740,1760],columns=['SAT'])
print(new_data)
new_data['Predicted_GPA'] = reg.predict(new_data)
print(new_data)

# Creating Plot
plt.scatter(x,y)
yhat = reg.coef_ * x_matrix + reg.intercept_
fig = plt.plot(x, yhat, lw=4, c='orange', label='regression line')
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)
plt.show()


