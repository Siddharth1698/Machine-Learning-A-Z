import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values  # left on the lines is lines of dataset and right of line is coloumns
Y = dataset.iloc[:,3].values