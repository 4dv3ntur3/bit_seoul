#2020-11-18 (8일차)
#Boston -> DNN + ModelCheckPoint
#흑백


from numpy import array
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input


#1. 데이터
from sklearn.datasets import load_boston
dataset = load_boston() #data(X)와 target(Y)으로 구분되어 있다
x = dataset.data
y = dataset.target



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


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#========= 1. load_model (fit 이후 save 모델) ===================
#3. 컴파일, 훈련

from tensorflow.keras.models import load_model
model1 = load_model('./save/boston_dnn_model_weights.h5')

#4. 평가, 예측
# result1 = model1.evaluate(x_test, y_test, batch_size=32)
# print("====model & weights 같이 저장=========")
# print("loss : ", result1[0])
# print("accuracy : ", result1[1])



y_pred_1 = model1.predict(x_test)

#RMSE


def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))


    # predict해서 나온 값과 원래 y_test 값을 비교해서 RMSE로 나오게 하겠다
print("RMSE 1: ", RMSE(y_test, y_pred_1))


# R2는 함수 제공

print("R2 1: ", r2_score(y_test, y_pred_1))




############## 2. load_model ModelCheckPoint #############

from tensorflow.keras.models import load_model
model2 = load_model('./model/boston_dnn-24-7.9741.hdf5')

#4. 평가, 예측

# result2 = model2.evaluate(x_test, y_test, batch_size=32)
# print("=======checkpoint 저장=========")
# print("loss : ", result2[0])
# print("accuracy : ", result2[1])



y_pred_2 = model2.predict(x_test)

#RMSE


print("RMSE 2: ", RMSE(y_test, y_pred_2))


# R2는 함수 제공

print("R2 2: ", r2_score(y_test, y_pred_2))




################ 3. load_weights ##################


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout #LSTM도 layer


model3 = Sequential()
model3.add(Dense(64, activation='relu', input_shape=(13,))) #default activation = linear

#hidden layer
model3.add(Dense(256, activation='relu'))
model3.add(Dense(400, activation='relu'))
model3.add(Dense(512, activation='relu'))
model3.add(Dense(1500, activation='relu'))
model3.add(Dense(750, activation='relu'))
model3.add(Dense(400, activation='relu'))
model3.add(Dense(150, activation='relu'))
model3.add(Dense(64, activation='relu'))
model3.add(Dense(1)) #output


# model.summary()

# 3. 컴파일

model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model3.load_weights('./save/boston_dnn_weights.h5')


#4. 평가, 예측
# result3 = model3.evaluate(x_test, y_test, batch_size=32)
# print("========weights 저장=========")
# print("loss : ", result3[0])
# print("accuracy : ", result3[1])



y_pred_3 = model3.predict(x_test)

#RMSE

    # predict해서 나온 값과 원래 y_test 값을 비교해서 RMSE로 나오게 하겠다
print("RMSE 3: ", RMSE(y_test, y_pred_3))


# R2는 함수 제공

print("R2 3: ", r2_score(y_test, y_pred_3))





'''
RMSE 1:  2.872187150108029
R2 1:  0.920124231075729


RMSE 2:  3.528238261208967
R2 2:  0.87946715912409


RMSE 3:  2.872187150108029
R2 3:  0.920124231075729
'''