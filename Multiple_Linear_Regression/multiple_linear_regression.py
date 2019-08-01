# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
from sklearn.compose import ColumnTransformer
     
ct = ColumnTransformer(
[('one_hot_encoder', OneHotEncoder(), [3])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
remainder='passthrough'                         # Leave the rest of the columns untouched
)
     
X = np.array(ct.fit_transform(X), dtype=np.float)

X = X[:,1:]



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)