#2020-11-19 (9일차)
#dataset *.npy로 저장하기: cifar10, fashion, cifar100, boston, diabetes, cancer

import numpy as np
from tensorflow.keras.datasets import cifar10, cifar100, fashion_mnist
from sklearn.datasets import load_boston, load_diabetes, load_breast_cancer



#1. cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
np.save('./data/cifar10_x_train.npy', arr=x_train)
np.save('./data/cifar10_y_train.npy', arr=y_train)
np.save('./data/cifar10_x_test.npy', arr=x_test)
np.save('./data/cifar10_y_test.npy', arr=y_test)


#2. fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
np.save('./data/fashion_x_train.npy', arr=x_train)
np.save('./data/fashion_y_train.npy', arr=y_train)
np.save('./data/fashion_x_test.npy', arr=x_test)
np.save('./data/fashion_y_test.npy', arr=y_test)


#3. cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
np.save('./data/cifar100_x_train.npy', arr=x_train)
np.save('./data/cifar100_y_train.npy', arr=y_train)
np.save('./data/cifar100_x_test.npy', arr=x_test)
np.save('./data/cifar100_y_test.npy', arr=y_test)


#4. boston
dataset = load_boston()
x = dataset.data
y = dataset.target
np.save('./data/boston_x.npy', arr=x)
np.save('./data/boston_y.npy', arr=y)


#5. diabetes
dataset = load_diabetes()
x = dataset.data
y = dataset.target
np.save('./data/diabetes_x.npy', arr=x)
np.save('./data/diabetes_y.npy', arr=y)


#6. breast_cancer
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
np.save('./data/cancer_x.npy', arr=x)
np.save('./data/cancer_y.npy', arr=y)







