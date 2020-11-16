<<<<<<< HEAD
#2020-11-10 (2일차)
#data slicing
#주어진 데이터에서 python 리스트 slicing으로 훈련/검증/평가 데이터 분리하기


#1. 데이터
import numpy as np

x = np.array(range(1, 101))
y = np.array(range(101, 201))
# weight = 1, bias = 100
# validation split = 0.2

x_train = np.array(x[:60]) # 60 
y_train = np.array(y[:60])

x_val = np.array(x[60:80])
y_val = np.array(y[60:80])

x_test = np.array(x[80:]) 
y_test = np.array(y[80:])


#2. 모델 구현
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(300, input_dim=1))

model.add(Dense(50)) 
model.add(Dense(70))
model.add(Dense(90))
model.add(Dense(10))
model.add(Dense(70))
model.add(Dense(50))

model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x_train, y_train, batch_size=1, epochs=500, validation_data=(x_val, y_val))


#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE(y_test, y_predict))


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2: ", r2)
=======
#2020-11-10 (2일차)
#data slicing
#주어진 데이터에서 python 리스트 slicing으로 훈련/검증/평가 데이터 분리하기


#1. 데이터
import numpy as np

x = np.array(range(1, 101))
y = np.array(range(101, 201))
# weight = 1, bias = 100
# validation split = 0.2

x_train = np.array(x[:60]) # 60 
y_train = np.array(y[:60])

x_val = np.array(x[60:80])
y_val = np.array(y[60:80])

x_test = np.array(x[80:]) 
y_test = np.array(y[80:])


#2. 모델 구현
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(300, input_dim=1))

model.add(Dense(50)) 
model.add(Dense(70))
model.add(Dense(90))
model.add(Dense(10))
model.add(Dense(70))
model.add(Dense(50))

model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x_train, y_train, batch_size=1, epochs=500, validation_data=(x_val, y_val))


#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE(y_test, y_predict))


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2: ", r2)
>>>>>>> b4eac5d0e44c2b94dcc0999e9053d1954a85c531
