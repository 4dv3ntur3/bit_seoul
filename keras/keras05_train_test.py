#2020-11-10 (2일차)
#훈련시킨 데이터로 평가하면 x

import numpy as np

#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_pred = np.array([11, 12, 13])

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense


#2. 모델 구성 
model = Sequential() 

model.add(Dense(300, input_dim=1)) # input_dim & 마지막 dense 동일해야

# hidden
model.add(Dense(500)) 
model.add(Dense(30))
model.add(Dense(70))
# layers

model.add(Dense(1))


#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam', metrics=['mae']) 

#model.fit(x, y, epochs=100, batch_size=1)
model.fit(x, y, epochs=1100) 


#4. 평가, 예측
# loss, acc = model.evaluate(x, y, bacth_size=1)
# loss, acc = model.evaluate(x, y)
loss= model.evaluate(x, y)

print("loss : ", loss)
# print("acc : ", acc)

y_pred = model.predict(x_pred)
print("결과물 : \n : ", y_pred)