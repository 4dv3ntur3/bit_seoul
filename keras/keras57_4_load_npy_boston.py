#2020-11-19 (9일차)
#*.npy로 저장한 dataset 불러오기: boston


#1. 데이터
import numpy as np
from numpy import array

x = np.load('./data/boston_x.npy')
y = np.load('./data/boston_y.npy')




#전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7
)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))



from tensorflow.keras.models import load_model
############### 1. load_model (fit 이후 save 모델) ##############
#3. 컴파일, 훈련

model1 = load_model('./save/boston_dnn_model_weights.h5')

#4. 평가, 예측
print("====model & weights 같이 저장=========")
y_pred_1 = model1.predict(x_test)
print("RMSE 1: ", RMSE(y_test, y_pred_1))
print("R2 1: ", r2_score(y_test, y_pred_1))




############## 2. load_model ModelCheckPoint #############

model2 = load_model('./model/boston_dnn-24-7.9741.hdf5')

#4. 평가, 예측
print("=======checkpoint 저장=========")
y_pred_2 = model2.predict(x_test)
print("RMSE 2: ", RMSE(y_test, y_pred_2))
print("R2 2: ", r2_score(y_test, y_pred_2))




################ 3. load_weights ##################
#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout #LSTM도 layer


model3 = Sequential()
model3.add(Dense(64, activation='relu', input_shape=(13,)))

#hidden layer
model3.add(Dense(256, activation='relu'))
model3.add(Dense(400, activation='relu'))
model3.add(Dense(512, activation='relu'))
model3.add(Dense(1500, activation='relu'))
model3.add(Dense(750, activation='relu'))
model3.add(Dense(400, activation='relu'))
model3.add(Dense(150, activation='relu'))
model3.add(Dense(64, activation='relu'))
model3.add(Dense(1))


# 3. 컴파일
model3.compile(loss='mse', optimizer='adam', metrics=['mse'])
model3.load_weights('./save/boston_dnn_weights.h5')


#4. 평가, 예측
print("========weights 저장=========")
y_pred_3 = model3.predict(x_test)
print("RMSE 3: ", RMSE(y_test, y_pred_3))
print("R2 3: ", r2_score(y_test, y_pred_3))

