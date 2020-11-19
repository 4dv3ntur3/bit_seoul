#2020-11-19 (9일차)
#Conv1D: 2차원 data, 1차원 input_shape
#Conv2D와 사용법은 동일

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Dropout
from tensorflow.keras.layers import MaxPooling1D


#1. 데이터
a = np.array(range(1, 101))
size = 5 #4개는 x, 1개는 y

#split 함수
def split_x(seq, size):
    aaa = [] #는 테스트
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size)]
        aaa.append(subset)
         
    # print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)

x = dataset[:, :4] #(95, 4)
y = dataset[:, 4] # (95, )

x = x.reshape(x.shape[0], x.shape[1], 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7
)

x_predict = np.array([97, 98, 99, 100])
x_predict = x_predict.reshape(1, 4, 1)


#conv1D 모델 구성
#2차원에서는 가로x세로, 1차원의 경우 쭉이니까 스칼라로, stride 있음, padding 있음
model = Sequential()
model.add(Conv1D(32, 3, activation='relu', padding='same', input_shape=(x.shape[1], x.shape[2])))
# model.add(Conv1D(32, 3, activation='relu', padding='same'))
# model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(64, 3, padding='same', activation='relu'))
# model.add(Conv1D(64, 3, activation='relu', padding='same'))
# model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(128, 3, padding='same', activation='relu'))
# model.add(Conv1D(128, 3, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())
model.add(Dense(300, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(1)) #ouput 


#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping

model.compile(loss='mse', optimizer='adam', metrics='mse')

early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping]
)


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)

y_predict = model.predict(x_predict)

model.summary()


print("loss: ", loss)

print("예측값: ", y_predict)


'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv1d (Conv1D)              (None, 4, 32)             128
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 4, 64)             6208
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 4, 128)            24704
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 2, 128)            0
_________________________________________________________________
flatten (Flatten)            (None, 256)               0
_________________________________________________________________
dense (Dense)                (None, 300)               77100
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 301
=================================================================
Total params: 108,441
Trainable params: 108,441
Non-trainable params: 0
_________________________________________________________________
loss:  0.008320051245391369
예측값:  [[101.048355]]
'''