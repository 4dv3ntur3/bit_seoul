#2020-11-19 (9일차)
#dataset *.npy로 저장하기: iris

import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
print(iris)
print(type(iris)) #<class 'sklearn.utils.Bunch'> 묶어서 준다. 갈라서 써야 함.

x_data = iris.data
y_data = iris.target


#데이터 형식 확인

print(type(x_data)) #<class 'numpy.ndarray'>
print(type(y_data)) #<class 'numpy.ndarray'>


#numpy로 저장해 보기
#경로, x_data
np.save('./data/iris_x.npy', arr=x_data)
np.save('./data/iris_y.npy', arr=y_data)