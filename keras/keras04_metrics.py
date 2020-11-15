#2020-11-10 (2일차)
#metrcis 없어도 돌아간다


import numpy as np


#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense


#2. 모델 구성 
model = Sequential() 


model.add(Dense(30, input_dim=1)) 
model.add(Dense(50))  
model.add(Dense(30))
model.add(Dense(7))
model.add(Dense(1))


#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam') #metrics=['acc']


#model.fit(x, y, epochs=100, batch_size=1)
model.fit(x, y, epochs=100) 


#4. 평가, 예측
# loss, acc = model.evaluate(x, y, bacth_size=1)
# loss, acc = model.evaluate(x, y)
loss= model.evaluate(x, y)

print("loss : ", loss)
# print("acc : ", acc)

# y_pred = model.predict(x)

# print("결과물 : \n : ", y_pred)