from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt

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
# ຂໍ້ມູນ 1 ຄ່າ -> class 0 ຫຼືບໍ? True : False
predict = 5000
y_train_0 = (y_train == 0)
