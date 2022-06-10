import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("https://raw.githubusercontent.com/kongruksiamza/MachineLearning/master/Linear%20Regression/Weather.csv")


# print(dataset.describe())
# dataset.plot(x='MinTemp', y='MaxTemp',style='o')
# plt.title('Min & Max Temp')
# plt.xlabel('Mintemp')
# plt.ylabel('Maxtemp')
# plt.show()

# train & test set
x = dataset['MinTemp'].values.reshape(-1, 1)
y = dataset['MaxTemp'].values.reshape(-1, 1)

# 80% - 20%
x_trian, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# training
model = LinearRegression()
model.fit(x_trian, y_train)

# test
y_pred = model.predict(x_test)

# plt.scatter(x_test, y_test)
# plt.plot(x_test, y_pred, color='red', linewidth=2)
# plt.show()

# compare true data & predict data
df = pd.DataFrame({'Actually': y_test.flatten(),
				  'Predicted': y_pred.flatten()})

# print(df.head())

df1 = df.head(20)
df1.plot(kind='bar', figsize=(16, 10))
plt.show()
