#2020-11-18 (8일차)
#diabetes -> DNN: load_model
#1) 모델&가중치(fit 이후 save한 모델 / ModelCheckPoint / 가중치)


from numpy import array
import numpy as np


#1. 데이터
from sklearn.datasets import load_diabetes
dataset = load_diabetes() #data(X)와 target(Y)으로 구분되어 있다
x = dataset.data #(442, 10)
y = dataset.target #(442, )



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=33 #shuffle 고정
)


#전처리
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    # predict해서 나온 값과 원래 y_test 값을 비교해서 RMSE로 나오게 하겠다



#========= 1. load_model (fit 이후 save 모델) ===================
#3. 컴파일, 훈련

from tensorflow.keras.models import load_model
model1 = load_model('./save/diabetes_dnn_model_weights.h5')

#4. 평가, 예측
print("====model & weights 같이 저장=========")
y_pred_1 = model1.predict(x_test)
print("RMSE 1: ", RMSE(y_test, y_pred_1))
print("R2 1: ", r2_score(y_test, y_pred_1))




############## 2. load_model ModelCheckPoint #############

model2 = load_model('./model/diabetes_DNN-41-3641.0332.hdf5')

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
model3.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model3.add(Dense(256, activation='relu'))
model3.add(Dense(128, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(100, activation='relu'))
model3.add(Dense(64, activation='relu'))
model3.add(Dense(32, activation='relu'))
model3.add(Dense(10, activation='relu'))
model3.add(Dense(1))


# 3. 컴파일
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model3.load_weights('./save/diabetes_dnn_weights.h5')


#4. 평가, 예측
print("========weights 저장=========")
y_pred_3 = model3.predict(x_test)
print("RMSE 3: ", RMSE(y_test, y_pred_3))
print("R2 3: ", r2_score(y_test, y_pred_3))


'''
RMSE 1:  52.87303868763434
R2 1:  0.4871686313608924
RMSE 2:  51.60574131434934
R2 2:  0.5114578010076998
RMSE 3:  52.87303868763434
R2 3:  0.4871686313608924
'''