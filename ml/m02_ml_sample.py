#2020-11-23 (11일차)
#Machine Learning
#iris_DNN copy - Onehotencoding, early_stopping, layer x

import numpy as np
from sklearn.datasets import load_iris

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D, Dropout


##### ML
from sklearn.svm import LinearSVC


#####1. 데이터 
#데이터 구조 확인
dataset = load_iris() #data(X)와 target(Y)으로 구분되어 있다

x, y = load_iris(return_X_y=True) #자동으로 x, y 나눠서 언패킹
# x = dataset.data
# y = dataset.target


# print(x.shape) #(150, 4)
# print(y.shape) #(150,)



from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, #random_state = 66, train_size = 0.8
)


model = LinearSVC() #_SVC: 분류모델에서 사용 (선형분류?)

model.fit(x_train, y_train)


#####4. 평가, 예측

#evaluate 대신 score
result = model.score(x_test, y_test)
print("score: ", result)




'''
Machine Learning
score:  0.9777777777777777



=======iris_dnn=======
Model: "sequential"
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 64)                320
_________________________________________________________________
dense_1 (Dense)              (None, 512)               33280
_________________________________________________________________
dropout (Dropout)            (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 256)               131328
_________________________________________________________________
dense_3 (Dense)              (None, 128)               32896
_________________________________________________________________
dense_4 (Dense)              (None, 300)               38700
_________________________________________________________________
dense_5 (Dense)              (None, 150)               45150
_________________________________________________________________
dense_6 (Dense)              (None, 70)                10570
_________________________________________________________________
dense_7 (Dense)              (None, 3)                 213
=================================================================
Total params: 292,457
Trainable params: 292,457
Non-trainable params: 0
_________________________________________________________________
loss:  0.11479297280311584
acc:  0.9777777791023254
예측값:  [1 0 1 1 0 2 2 1 1 2]
정답:  [1 0 1 1 0 1 2 1 2 2]
'''

