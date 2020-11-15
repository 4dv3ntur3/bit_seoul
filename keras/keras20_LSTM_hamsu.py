#2020-11-12 (4일차)
#RNN(Recurrent Neural Network): LSTM, SimpleRNN, GRU
#LSTM 함수형 모델로



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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input


#함수형 모델
input1 = Input(shape=(3,1))
dense1 = LSTM(30, activation='relu')(input1)
dense2 = Dense(70, activation='relu')(dense1)
dense3 = Dense(100, activation='relu')(dense2)
dense4 = Dense(50, activation='relu')(dense3)
dense5 = Dense(30, activation='relu')(dense4)
dense6 = Dense(10, activation='relu')(dense5)
output1 = Dense(1)(dense6)

model = Model(inputs=input1, outputs=output1)


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