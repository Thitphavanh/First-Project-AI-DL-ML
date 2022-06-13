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
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

# 0 - 9
# 5000
# 5
# True, False
