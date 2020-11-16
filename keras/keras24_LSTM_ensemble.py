<<<<<<< HEAD
#2020-11-12 (4일차)
#RNN(Recurrent Neural Network): LSTM, SimpleRNN, GRU
#시계열 데이터의 경우 가장 영향을 크게 끼치는 데이터는 근접해 있는 데이터다 

#함수형으로 LSTM 모델 두 개 만들어서 앙상블
#X1, X2 -> Y

import numpy as np
from numpy import array



#1. 데이터
x1 = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], 
           [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
           [9, 10, 11], [10, 11, 12],
           [20, 30, 40], [30, 40, 50], [40, 50, 60]])


x2 = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60], 
           [50, 60, 70], [60, 70, 80], [70, 80, 90], [80, 90, 100],
           [90, 100, 110], [100, 110, 120],
           [2, 3, 4], [3, 4, 5], [4, 5, 6]])

y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])

x1_predict = array([55, 65, 75])
x2_predict = array([65, 75, 85])

x1 = x1.reshape(13, 3, 1)
x2 = x2.reshape(13, 3, 1)

x1_predict = x1_predict.reshape(1, 3, 1)
x2_predict = x2_predict.reshape(1, 3, 1)

#2. 모델 구성
#import 빠뜨린 거 없이 할 것
from tensorflow.keras.models import Model, Sequential 
from tensorflow.keras.layers import Dense, LSTM, Input

#모델 1
input1 = Input(shape=(3,1))
dense1_1 = LSTM(200, activation='relu')(input1)
dense1_2 = Dense(180, activation='relu')(dense1_1)
dense1_3 = Dense(150, activation='relu')(dense1_2)
dense1_4 = Dense(30, activation='relu')(dense1_3)
dense1_5 = Dense(10, activation='relu')(dense1_4)
output1 = Dense(1)(dense1_5)
model1 = Model(inputs=input1, outputs=output1)



#85.07, 95.53
#모델2
input2 = Input(shape=(3,1))
dense2_1 = LSTM(200, activation='relu')(input2)
dense2_2 = Dense(180, activation='relu')(dense2_1)
dense2_3 = Dense(150, activation='relu')(dense2_2)
dense2_4 = Dense(30, activation='relu')(dense2_3)
dense2_5 = Dense(10, activation='relu')(dense2_4)
output2 = Dense(1)(dense2_5)
model2 = Model(inputs=input2, outputs=output2)


#병합 
from tensorflow.keras.layers import Concatenate, concatenate

merge1 = Concatenate()([output1, output2])
middle1 = Dense(30)(merge1)
middle1 = Dense(70)(middle1)
output1 = Dense(200)(middle1)
output1 = Dense(50)(output1)
output1 = Dense(30)(output1)
output1 = Dense(1)(output1)

model = Model(inputs=[input1, input2], outputs=output1)


#3. 컴파일, 훈련

#early stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=85, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(
    [x1, x2],
    y,
    callbacks=[early_stopping],
    epochs=1000, batch_size=1
)

#4. 평가
# result = model.evaluate([x1, x2], y)s


#예측값은 모델에 맞게 넣어주되 predict를 50번 하든 몇 번 하든 원하는 결과만 내면 됨
y_pred_1 = model.predict([x1_predict, x2_predict])
y_pred_2 = model.predict([x2_predict, x1_predict])

print(y_pred_1)
=======
#2020-11-12 (4일차)
#RNN(Recurrent Neural Network): LSTM, SimpleRNN, GRU
#시계열 데이터의 경우 가장 영향을 크게 끼치는 데이터는 근접해 있는 데이터다 

#함수형으로 LSTM 모델 두 개 만들어서 앙상블
#X1, X2 -> Y

import numpy as np
from numpy import array



#1. 데이터
x1 = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], 
           [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
           [9, 10, 11], [10, 11, 12],
           [20, 30, 40], [30, 40, 50], [40, 50, 60]])


x2 = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60], 
           [50, 60, 70], [60, 70, 80], [70, 80, 90], [80, 90, 100],
           [90, 100, 110], [100, 110, 120],
           [2, 3, 4], [3, 4, 5], [4, 5, 6]])

y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])

x1_predict = array([55, 65, 75])
x2_predict = array([65, 75, 85])

x1 = x1.reshape(13, 3, 1)
x2 = x2.reshape(13, 3, 1)

x1_predict = x1_predict.reshape(1, 3, 1)
x2_predict = x2_predict.reshape(1, 3, 1)

#2. 모델 구성
#import 빠뜨린 거 없이 할 것
from tensorflow.keras.models import Model, Sequential 
from tensorflow.keras.layers import Dense, LSTM, Input

#모델 1
input1 = Input(shape=(3,1))
dense1_1 = LSTM(200, activation='relu')(input1)
dense1_2 = Dense(180, activation='relu')(dense1_1)
dense1_3 = Dense(150, activation='relu')(dense1_2)
dense1_4 = Dense(30, activation='relu')(dense1_3)
dense1_5 = Dense(10, activation='relu')(dense1_4)
output1 = Dense(1)(dense1_5)
model1 = Model(inputs=input1, outputs=output1)



#85.07, 95.53
#모델2
input2 = Input(shape=(3,1))
dense2_1 = LSTM(200, activation='relu')(input2)
dense2_2 = Dense(180, activation='relu')(dense2_1)
dense2_3 = Dense(150, activation='relu')(dense2_2)
dense2_4 = Dense(30, activation='relu')(dense2_3)
dense2_5 = Dense(10, activation='relu')(dense2_4)
output2 = Dense(1)(dense2_5)
model2 = Model(inputs=input2, outputs=output2)


#병합 
from tensorflow.keras.layers import Concatenate, concatenate

merge1 = Concatenate()([output1, output2])
middle1 = Dense(30)(merge1)
middle1 = Dense(70)(middle1)
output1 = Dense(200)(middle1)
output1 = Dense(50)(output1)
output1 = Dense(30)(output1)
output1 = Dense(1)(output1)

model = Model(inputs=[input1, input2], outputs=output1)


#3. 컴파일, 훈련

#early stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=85, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(
    [x1, x2],
    y,
    callbacks=[early_stopping],
    epochs=1000, batch_size=1
)

#4. 평가
# result = model.evaluate([x1, x2], y)s


#예측값은 모델에 맞게 넣어주되 predict를 50번 하든 몇 번 하든 원하는 결과만 내면 됨
y_pred_1 = model.predict([x1_predict, x2_predict])
y_pred_2 = model.predict([x2_predict, x1_predict])

print(y_pred_1)
>>>>>>> b4eac5d0e44c2b94dcc0999e9053d1954a85c531
print(y_pred_2)