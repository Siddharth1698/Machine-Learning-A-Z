
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.ensemble import RandomForestRegressor
# n esitimator -> no of trees used
regressor = RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(X,y)

y_pred = regressor.predict (np.array([6.5]).reshape (-1, 1))

# Visualising the Random Forest Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()