#2020-11-17 (7일차)
#Diatbetes -> LSTM
#사이킷런의 dataset


from numpy import array
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input


#1. 데이터
from sklearn.datasets import load_diabetes
dataset = load_diabetes() #data(X)와 target(Y)으로 구분되어 있다
x = dataset.data
y = dataset.target

# print("x: ", x)
# print("y: ", y) #전처리가 되어 있지 않다

# print(x.shape) #(506, 13)
# print(y.shape) #(506,)


#전처리: 수치가 크다

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7
)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
#train

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# x_pred = scaler.transform(x_pred)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout #LSTM도 layer


model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(x_train.shape[1], 1))) #default activation = linear

#hidden layer
model.add(Dense(256, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1500, activation='relu'))
model.add(Dense(750, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1)) #output




#3. 컴파일 및 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(
    x_train,
    y_train,
    callbacks=[early_stopping],
    validation_split=0.2,
    epochs=100, batch_size=32
)



#4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=32)

print("======Diabetes_LSTM=====")
model.summary()

print("loss, mse: ", loss, mse)


y_pred = model.predict(x_test)


#RMSE
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_pred))
    # predict해서 나온 값과 원래 y_test 값을 비교해서 RMSE로 나오게 하겠다
print("RMSE: ", RMSE(y_test, y_pred))


# R2는 함수 제공
from sklearn.metrics import r2_score
print("R2: ", r2_score(y_test, y_pred))