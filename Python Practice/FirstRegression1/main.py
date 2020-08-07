import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()  # overrides style and graphic on mathplotlib
# it's a pretty skin for matplotlib

# convert data variable into data frame
data = pd.read_csv('resources/1.01. Simple linear regression.csv')

print(data)

# give the most useful statistics
print(data.describe())

# Regression should be meaningful
# Regression equation is y = b0 + b1x1
y = data['GPA']  # this is our dependent variable 'GPA'
x1 = data['SAT']  # independent variable

plt.scatter(x1, y)
plt.title('SAT and GPA Correlation')
plt.xlabel('SAT', fontsize='20')
plt.ylabel('GPA', fontsize='20')
plt.show()

x = sm.add_constant(x1)
results = sm.OLS(y, x).fit()  # OLS = ordinary least squares development
# OLS is the most common method for regression. Gets minimum SSE (Sum of Squared Error)
# other methods to get regression line
#       Generalized Least Squares
#       Maximum likelihood estimation
#       Bayesian regression
#       Kernel regression
#       Gaussian process regression
print(results.summary())

# Now adding regression line
plt.scatter(x1, y)
plt.title('SAT and GPA Correlation With Regression Line')
yhat = 0.0017 * x1 + 0.275
fig = plt.plot(x1, yhat, lw=4, c='orange', label='regression line')
plt.xlabel('SAT', fontsize='20')
plt.ylabel('GPA', fontsize='20')
plt.show()

# Adjusted R-Squared

data = pd.read_csv('resources/1.02. Multiple linear regression.csv')
# from data college GPA = b0 + b1*SAT + b2*Rand1,2,3
y = data['GPA']
x1 = data[['SAT', 'Rand 1,2,3']]
x = sm.add_constant(x1)
results = sm.OLS(y, x).fit()
print(results.summary())
# Rand 1,2,3 is a useless variable, we need to drop it

# replacing categories with number
raw_data = pd.read_csv('resources/1.03. Dummies.csv')
data = raw_data.copy()
data['Attendance'] = data['Attendance'].map({'Yes': 1, 'No': 0})

print(data.describe())
y = data['GPA']
x1 = data[['SAT','Attendance']]
x = sm.add_constant(x1)
results = sm.OLS(y, x).fit()
print(results.summary())






