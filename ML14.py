from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

#  Read Data
df = pd.read_csv('diabetes.csv')

# Data
x = df.drop('Outcome', axis=1).values

#  Outcome Data
y = df['Outcome'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

# find k to model
k = np.arange(1, 9)

