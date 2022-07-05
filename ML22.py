from turtle import color
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.svm import SVC
from sklearn.model_selection import *


# download & display image
faces = fetch_lfw_people(min_faces_per_person=60)
# print(faces.target_names)
# print(faces.images.shape)


'''
fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[faces.target[i]].split()[-1], color='black')
plt.show()
'''

# reduce & create model
pca = PCA(n_components=150, svd_solver='randomized', whiten=True)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)

# train, test data
x_train, x_test, y_train, y_test = train_test_split(faces.data, faces.target, random_state=40)
param = {'svc__C': [1, 5, 10, 50], 'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}

# train data to model
grid = GridSearchCV(model, param)
grid.fit(x_train, y_train)
# print(grid.best_estimator_)

model=grid.best_estimator_
# predict