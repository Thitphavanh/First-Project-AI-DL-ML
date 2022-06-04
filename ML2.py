from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()

# (150,4)
# 75%, 25%
x_trian, x_test, y_trian, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], test_size=0.2,
    random_state=0)

print(x_trian.shape)
print(x_test.shape)

print(y_trian.shape)
print(y_test.shape)

# 150
# train 80 % = 120
# test 2 % = 30
