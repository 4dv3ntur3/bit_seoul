
# Multi Layer Perceptron

# 1 -> 3

#output layer를 결과랑 맞춰줘야함 (y dims랑)
#1. 데이터

import numpy as np

x = np.array([range(1, 101)])
y = np.array([range(101, 201), range(311, 411), range(100)])


x = x.reshape(100,)
y = y.transpose()


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6)



# slicing
# x_train = x[:60]
# y_train = y[:60]
# x_test = x[60:80]
# y_test = y[60:80]



#확인용

#for i in range(0, 50):
# print(i+1, ". ", "x: ", x[i], "\n", "     y: ", y[i])

#print(x)
#print(y)

#print(x.shape)
#print(y.shape)


#행 무시, 열 우선
#data의 특성은 "열"에 의해 결정된다
#특성=feature=column=열



#최종목표: y1, y2, y3 = w1x1 + w2x2 + w3x3 + b (y도 3개)

#2. 모델 구성
from tensorflow.keras.models import Sequential #순차적 모델
from tensorflow.keras.layers import Dense #DNN 구조의 Dense모델

model = Sequential()
model.add(Dense(300, input_dim=1)) #column개수=3개
model.add(Dense(50))
model.add(Dense(70))
model.add(Dense(90))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(3)) # 출력 =3개


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=150, batch_size=1, validation_split=0.2)

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