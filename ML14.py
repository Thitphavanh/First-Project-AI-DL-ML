from cProfile import label
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#  Read Data
df = pd.read_csv('diabetes.csv')

# Data
x = df.drop('Outcome', axis=1).values

#  Outcome Data
y = df['Outcome'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

# find k to model
k_neighbors = np.arange(1, 9)
# empty
train_score = np.empty(len(k_neighbors))
test_score = np.empty(len(k_neighbors))

for i, k in enumerate(k_neighbors):
    # 1,2,3,4,5,6,7,8
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    # ກວດສອບປະສິດທິພາບ
    train_score[i] = knn.score(x_train, y_train)
    test_score[i] = knn.score(x_test, y_test)
    print(test_score[i]*100)

plt.title('Compare k value in model')
plt.plot(k_neighbors, test_score, label='Test Score')
plt.plot(k_neighbors, train_score, label='Train Score')
plt.legend()
plt.xlabel('k Number')
plt.ylabel('Score')
plt.show()
