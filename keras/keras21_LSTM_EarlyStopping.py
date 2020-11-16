#2020-11-12 (4일차)
#RNN(Recurrent Neural Network): LSTM, SimpleRNN, GRU
#early stopping: monitor, patience, mode


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
dense4 = Dense(30, activation='relu')(dense3)
dense5 = Dense(10, activation='relu')(dense4)
output1 = Dense(1)(dense5)

model = Model(inputs=input1, outputs=output1)



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='mse')


#import 뭘 할지 다 알고 있다면 미리 최상단에 기재하는 것이 좋음
from tensorflow.keras.callbacks import EarlyStopping

#early stopping이 무조건 좋지 않음! hyper parameter tuning 부분 추가!
# early_stopping = EarlyStopping(monitor='loss', patience=100, mode='min') 

#80~90사이 놓고 돌려 보기
early_stopping = EarlyStopping(monitor='loss', patience=85, mode='auto') #min/max 헷갈릴 때
#감시 기준을 loss, 그 최솟값에서 벗어나는 걸 몇 번 눈감아 줄 것인가, loss는 낮을수록 좋으므로 min
model.fit(x, y, epochs=10000, batch_size=1, callbacks=[early_stopping]) #훈련시킬 때마다 early_stopping 호출하겠다



#4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=1)
print("\nloss: ", loss)
#x_input reshape
x_input = x_input.reshape(1, 3, 1)
y_predict = model.predict(x_input)

print(y_predict)

model.summary()