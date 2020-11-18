#2020-11-18 (8일차)
#diabetes -> DNN + ModelCheckPoint
#사이킷런의 데이터셋 load_diabetes


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

# print(x.shape) #(442, 10)
# print(y.shape) #(442, )


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=33
)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# x_pred = scaler.transform(x_pred)





from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#========= 1. load_model (fit 이후 save 모델) ===================
#3. 컴파일, 훈련

from tensorflow.keras.models import load_model
model1 = load_model('./save/diabetes_dnn_model_weights.h5')

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
model2 = load_model('./model/diabetes_DNN-41-3641.0332.hdf5')

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
model3.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],))) #flatten하면서 곱하고 dense에서 또 100 곱함 
                                        #Conv2d의 activatio n default='relu'
                                        #LSTM의 activation default='tanh'
model3.add(Dense(256, activation='relu'))
model3.add(Dense(128, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(100, activation='relu'))
model3.add(Dense(64, activation='relu'))
model3.add(Dense(32, activation='relu'))
model3.add(Dense(10, activation='relu'))

#output _ 선형회귀 
model3.add(Dense(1))




# model.summary()

# 3. 컴파일

model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model3.load_weights('./save/diabetes_dnn_weights.h5')


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
RMSE 1:  52.87303868763434
R2 1:  0.4871686313608924
RMSE 2:  51.60574131434934
R2 2:  0.5114578010076998
RMSE 3:  52.87303868763434
R2 3:  0.4871686313608924
'''