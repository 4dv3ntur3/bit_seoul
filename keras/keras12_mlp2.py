#2020-11-11 (3일차)
#Multi Layer Perceptron
#data shape 맞추기: 행 무시, 열 우선
#transpose

#1. 데이터
import numpy as np

x = np.array((range(1, 101), range(711, 811), range(100)))
y = np.array(range(101, 201))


#데이터 shape 확인
print(x.shape)
print(y.shape)

#맞춰 주기
x = x.transpose()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)

#행 무시, 열 우선
#data의 특성은 "열"에 의해 결정된다
#특성=feature=column=열

#최종목표: y1, y2, y3 = w1x1 + w2x2 + w3x3 + b (y도 3개)

#2. 모델 구성
from tensorflow.keras.models import Sequential #순차적 모델
from tensorflow.keras.layers import Dense #DNN 구조의 Dense모델

model = Sequential()
model.add(Dense(100, input_dim=3)) #column개수=3개
model.add(Dense(50))
model.add(Dense(70))
model.add(Dense(90))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(1)) # output y=1이므로 노드 1개


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, validation_split=0.2)

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
