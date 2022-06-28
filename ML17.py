from cProfile import label
from matplotlib.pyplot import axis
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


def clean_data(dataset):
    for column in dataset.columns:
        if dataset[column].dtype == type(object):
            le = LabelEncoder()
            dataset[column] = le.fit_transform(dataset[column])
    return dataset


def split_feature_class(dataset, feature):
    features = dataset.drop(feature, axis=1)  # ເອົາຂໍ້ມູນທັງໝົດຍົກເວັ້ນ income
    labels = dataset[feature].copy()  # ເອົາສະເພາະຂໍ້ມູນ income
    return features, labels


dataset = pd.read_csv('adult.csv')
dataset = clean_data(dataset)


# split trian, test
train_set, test_set = train_test_split(dataset, test_size=0.2)

# train
train_features, train_lebels = split_feature_class(train_set, 'income')

# test
test_features, test_labels = split_feature_class(test_set, 'income')

# model
model = GaussianNB()
model.fit(train_features, train_lebels)

# predict
clf_pred = model.predict(test_features)

print('Accuracy =', accuracy_score(test_labels, clf_pred))
