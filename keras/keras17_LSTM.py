# 2020-11-12 

#1. 데이터
import numpy as np

#shape 표시 해주기
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5,], [4, 5, 6]]) #(4, 3)
y = np.array([4, 5, 6, 7])                                  #(4, ) = 스칼라 4개

# 123 -> 4, 234 -> 5, 345 -> 6, 456 -> 7


#항상 shape 찍어서 확인해 보기 
print("x.shape: ", x.shape)
print("y.shape: ", y.shape)

#shape 맞추기
x = x.reshape(x.shape[0], x.shape[1], 1)
#x = x.reshape(4, 3, 1)

print("x.shape: ", x.shape)


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM #LSTM도 layer

model = Sequential()
model.add(LSTM(20, activation='relu', input_shape=(3,1))) #이후부터는 동일한 dense층, 넘어가는 노드의 개수가 30개
model.add(Dense(50)) #default activation = linear
model.add(Dense(100))
model.add(Dense(30))
model.add(Dense(1)) #output: 1개

# model.summary()
# LSTM parameter = 4 * (몇개씩잘라서 + 1 (bias) + node수) * node수
#LSTM은 그만큼 연산이 많다 따라서 속도도 그만큼 느려짐

# LSTM은 input으로 2차원을 받는다 (DNN은 행 빼고 스칼라만 받고)
# LSTM의 shape => (행, 열, 몇 개씩 자르는지) 단, 전체 데이터 개수는 같게 reshape
# y값은 대부분 스칼라로 많이 나감. 그런데 LSTM은 parameter가 되게 많다(순환)
# 하지만 데이터 개수 적으니까 잘라서 하나씩! (3, 1) 즉, 1열씩 작업하겠다


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='mse')

model.fit(x, y, epochs=100, batch_size=1)
x_input = np.array([5, 6, 7]) #(3, ) -> (1, 3, 1) #LSTM에 들어가려면 맞춰 줘야 
#데이터 reshape
x_input = x_input.reshape(1, 3, 1)
#어느 정도 데이터 양도 필요하다 (LSTM 할 때)


#4. 평가, 예측
y_predict = model.predict(x_input)
loss, acc = model.evaluate(x, y, batch_size=1)

print("예측값: ", y_predict)
print("loss: ", loss, "\n", "acc: ", acc)



                




