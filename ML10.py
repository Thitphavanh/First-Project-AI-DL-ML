from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier


def displayImage(x):
    plt.imshow(x.reshape(28, 28), cmap=plt.cm.binary, interpolation='nearest')
    plt.show()


def displayPredict(clf, actually_y, x):
    print('Actually = ', actually_y)
    print('Prediction = ', clf.predict([x])[0])


mnist_raw = loadmat("mnist-original.mat")
mnist = {
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0]
}

# traing, set
# 1 - 60000
# 60001 - 70000
x, y = mnist["data"], mnist["target"]
# train & test set
# class 0 - 9
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

# 0 - 9
# 5000
# 5
# True, False

# class 0 ບໍ່ແມ່ນ class 0
# ຂໍ້ມູນຄ່າ 5000 -> model -> class 0 ຫຼືບໍ? True : False
# y_train = [0,0,........,9....,9]
predict_number = 500
y_train_0 = (y_train == 0)
y_test_0 = (y_test == 0)

# y_train_0 = [true,true,........,false....,false]


sgd_clf = SGDClassifier()
sgd_clf.fit(x_train, y_train_0)


displayPredict(sgd_clf, y_test_0[predict_number], x_test[predict_number])
displayImage(x_test[predict_number])
