from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size=0.4, random_state=0)


# Model
knn = KNeighborsClassifier(n_neighbors=1)


# training
knn.fit(x_train, y_train)

# prediction
pred = knn.predict([x_test[0]])

print('ຜົນການພະຍາກອນ', pred)
print('ທຳນານວ່າຢູ່ໃນກຸ່ມສາຍພັນ', iris_dataset['target_names'][pred])
