import numpy as np


#1. 데이터
#훈련할 때 쓴 데이터로 평가하는 건 답안지를 가지고 있는 것과 똑같다.
#그러므로 훈련 데이터(train), 훈련 후 평가할 데이터(test), 검증 데이터(val, 컨닝해서 참고할 데이터)로 분류한다.
# fit - train, val / evaluate - test

x = np.array(range(1, 101))
y = np.array(range(101, 201))

#train:test = 7:3
x_train = np.array(x[:70]) 
y_train = np.array(y[:70])
x_test = np.array(x[70:])
y_test = np.array(y[70:])

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

#input layer
model.add(Dense(300, input_dim=1))

#hidden 
model.add(Dense(50))
model.add(Dense(70))
model.add(Dense(90))
model.add(Dense(10))
model.add(Dense(70))
model.add(Dense(50))
#layers

#output layer
model.add(Dense(1)) #input_dim = output layer 노드 개수


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='Adam', metrics=['mae']) #metrics는 훈련에 영향을 주지 않음. 복수 옵션 입력 가능. acc는 선형회귀모델에 부적합.

model.fit(x, y, epochs=100, batch_size=1, validation_split=0.2) #batch_size의 기본값은 32(평타). 적절히 조절해야 함.
#검증 데이터 넣는 방법 2가지: 
# validation_data=(x_val, y_val) //// validation_split=0.2 (알아서 훈련 데이터에서 뺀다)

#4. 평가, 예측
loss, acc = model.evaluate(x_train, y_train, batch_size=1)
y_predict = model.predict(x_test)


# 평가 지표 활용을 위해 사이킷런 임포트
# RMSE(비교할 데이터 두 개 받아서 MSE 구하고 루트 씌우는 함수 만들기), R2
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2: ", r2)



