from sklearn import datasets
import matplotlib.pyplot as plt
digit_dataset = datasets.load_digits()

print(digit_dataset.target[0])
plt.imshow(digit_dataset.images[0], cmap=plt.get_cmap('gray'))
plt.show()
