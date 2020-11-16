<<<<<<< HEAD
#2020-11-12 (4일차)
#RNN(Recurrent Neural Network): LSTM, SimpleRNN, GRU
#LSTM vs Dense 

import numpy as np

#1. 데이터
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], 
             [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10], 
             [9, 10, 11], [10, 11 ,12],
             [20, 30, 40], [30, 40, 50], [40, 50, 60]])


y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])
x_input = np.array([50, 60, 70])




# print(x.shape)

#shape 맞추기
#LSTM을 사용하기 위해선 reshape가 필수불가결
#dense층에선 (13, 3)을 각 1열씩이라고 판단 가능하므로 reshape 필요 x

# x = x.reshape(13, 3, 1)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#80.02
model = Sequential()
model.add(Dense(30, activation='relu', input_dim=3)) #column 개수=3
model.add(Dense(70, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='mse')


#early stopping
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience=85, mode='auto') #min/max 헷갈릴 때
model.fit(x, y, epochs=1000, batch_size=1, callbacks=[early_stopping])


#4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=1)
print("\nloss: ", loss)


x_input = x_input.reshape(1, 3)
#x_input reshape
y_predict = model.predict(x_input)

print(y_predict)

# model.summary()
=======
#2020-11-12 (4일차)
#RNN(Recurrent Neural Network): LSTM, SimpleRNN, GRU
#LSTM vs Dense 

import numpy as np

#1. 데이터
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], 
             [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10], 
             [9, 10, 11], [10, 11 ,12],
             [20, 30, 40], [30, 40, 50], [40, 50, 60]])


y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])
x_input = np.array([50, 60, 70])




# print(x.shape)

#shape 맞추기
#LSTM을 사용하기 위해선 reshape가 필수불가결
#dense층에선 (13, 3)을 각 1열씩이라고 판단 가능하므로 reshape 필요 x

# x = x.reshape(13, 3, 1)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#80.02
model = Sequential()
model.add(Dense(30, activation='relu', input_dim=3)) #column 개수=3
model.add(Dense(70, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='mse')


#early stopping
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience=85, mode='auto') #min/max 헷갈릴 때
model.fit(x, y, epochs=1000, batch_size=1, callbacks=[early_stopping])


#4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=1)
print("\nloss: ", loss)


x_input = x_input.reshape(1, 3)
#x_input reshape
y_predict = model.predict(x_input)

print(y_predict)

# model.summary()
>>>>>>> b4eac5d0e44c2b94dcc0999e9053d1954a85c531
