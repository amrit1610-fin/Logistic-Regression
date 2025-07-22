import numpy as np
from LogisticRegression import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt

bc = datasets.load_breast_cancer()
X , y = bc.data , bc.target
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 1234) 

clf = LogisticRegression(lr=0.0001, n_iters=1000)
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)

def accuracy(y_pred , y_test):
    return np.sum(y_pred == y_test) / len(y_test)

acc = accuracy(y_pred , y_test)
print(acc)


