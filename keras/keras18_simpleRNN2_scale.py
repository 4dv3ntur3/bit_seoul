<<<<<<< HEAD
#2020-11-12 (4일차)
#RNN(Recurrent Neural Network): LSTM, SimpleRNN, GRU

# LSTM -> SimpleRNN (용법 같다)

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
x = x.reshape(13, 3, 1)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN


#실습: LSTM 완성
#예측값: 80

model = Sequential()
# model.add(LSTM(200, input_shape=(3, 1)))
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(30))
# model.add(Dense(10))
# model.add(Dense(1))


#과제: 80.00809 
model.add(SimpleRNN(30, activation='relu', input_shape=(3, 1))) 
model.add(Dense(70))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='mse')
model.fit(x, y, epochs=200, batch_size=1)


#4. 평가, 예측

loss, acc = model.evaluate(x, y, batch_size=1)
print("\nloss: ", loss)
#x_input reshape
x_input = x_input.reshape(1, 3, 1)
y_predict = model.predict(x_input)
print(y_predict)
model.summary()




# 동일한 모델로 돌렸을 경우
#------------------------------------------------------------------------------
#                       LSTM           |        simpleRNN           
#--------------------------------------------------------------------------------
# predict() |          80.300995       |        79.99036
#-------------------------------------------------------------------------------
# loss      |   0.8289151787757874     |     1.1793473277066369e-05
#-------------------------------------------------------------------------------
=======
#2020-11-12 (4일차)
#RNN(Recurrent Neural Network): LSTM, SimpleRNN, GRU

# LSTM -> SimpleRNN (용법 같다)

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
x = x.reshape(13, 3, 1)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN


#실습: LSTM 완성
#예측값: 80

model = Sequential()
# model.add(LSTM(200, input_shape=(3, 1)))
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(30))
# model.add(Dense(10))
# model.add(Dense(1))


#과제: 80.00809 
model.add(SimpleRNN(30, activation='relu', input_shape=(3, 1))) 
model.add(Dense(70))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='mse')
model.fit(x, y, epochs=200, batch_size=1)


#4. 평가, 예측

loss, acc = model.evaluate(x, y, batch_size=1)
print("\nloss: ", loss)
#x_input reshape
x_input = x_input.reshape(1, 3, 1)
y_predict = model.predict(x_input)
print(y_predict)
model.summary()




# 동일한 모델로 돌렸을 경우
#------------------------------------------------------------------------------
#                       LSTM           |        simpleRNN           
#--------------------------------------------------------------------------------
# predict() |          80.300995       |        79.99036
#-------------------------------------------------------------------------------
# loss      |   0.8289151787757874     |     1.1793473277066369e-05
#-------------------------------------------------------------------------------
>>>>>>> b4eac5d0e44c2b94dcc0999e9053d1954a85c531
# param     | LSTM:3840, total:20011   |    simpleRNN:960, total:17131