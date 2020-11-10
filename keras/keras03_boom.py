import numpy as np

#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense


#2. 모델 구성 
model = Sequential() 

model.add(Dense(3000000, input_dim=1)) 
model.add(Dense(5000000)) 
model.add(Dense(3000000))
model.add(Dense(700000))
model.add(Dense(100000))
model.add(Dense(1))


#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam', metrics=['acc']) # hyper parameter tuning


#model.fit(x, y, epochs=100, batch_size=1) # hyper parameter tuning
model.fit(x, y, epochs=10000) # hyper parameter tuning


#4. 평가, 예측
# loss, acc = model.evaluate(x, y, bacth_size=1)
loss, acc = model.evaluate(x, y)


print("loss : ", loss)
print("acc : ", acc)

y_pred = model.predict(x)

print("결과물 : \n : ", y_pred)