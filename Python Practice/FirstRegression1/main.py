import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# convert data variable into data frame
data = pd.read_csv('resources/1.01. Simple linear regression.csv')

print(data)

# give the most useful statistics
print(data.describe())

# Regression should be meaningful
# Regression equation is y = b0 + b1x1
y = data['GPA']  # this is our dependent variable 'GPA'
x1 = data['SAT']  # independent variable

plt.scatter(x1,y)
plt.title('SAT and GPA Correlation')
plt.xlabel('SAT', fontsize='20')
plt.ylabel('GPA', fontsize='20')
plt.show()

x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()  # OLS = ordinary least squares development
print(results.summary())


# Now adding regression line
plt.scatter(x1,y)
plt.title('SAT and GPA Correlation With Regression Line')
yhat = 0.0017*x1 + 0.275
fig = plt.plot(x1, yhat, lw=4, c = 'orange', label = 'regression line')
plt.xlabel('SAT', fontsize='20')
plt.ylabel('GPA', fontsize='20')
plt.show()