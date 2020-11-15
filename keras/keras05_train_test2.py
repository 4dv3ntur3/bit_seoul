#2020-11-10 (2일차)
#훈련시킬 데이터, 평가할 데이터 분리


import numpy as np

#1. 데이터

# _train : 훈련시킬 데이터
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# _test : 평가할 데이터(evaluate)
x_test = np.array([11, 12, 13, 14, 15])
y_test = np.array([11, 12, 13, 14, 15])

# _pred: 예측하고 싶은 데이터 (y값 없음. 모델링, 훈련을 바탕으로 예상해낸 y가 알고 싶은 것이므로)
x_pred = np.array([16, 17, 18])

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense


#2. 모델 구성 
model = Sequential() 

model.add(Dense(300, input_dim=1)) # input_dim & 마지막 dense 동일해야

# hidden
model.add(Dense(50)) 
model.add(Dense(70))
model.add(Dense(90))
model.add(Dense(10))
model.add(Dense(70))
model.add(Dense(50))
# layers

model.add(Dense(1))


#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam', metrics=['mae']) 

#model.fit(x, y, epochs=100, batch_size=1)
model.fit(x_train, y_train, epochs=1000) 


#4. 평가, 예측
# loss, acc = model.evaluate(x, y, bacth_size=1)
# loss, acc = model.evaluate(x, y)
loss = model.evaluate(x_test, y_test)

print("loss : ", loss)
# print("acc : ", acc)

y_pred = model.predict(x_pred)
print("결과물 : \n : ", y_pred)