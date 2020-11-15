#2020-11-09 (1일차)
#기본 코드


import numpy as np

#1. 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense #dense층


#2. 모델 구성 (node 쌓기. dense = 단순 DNN 층 구성)
model = Sequential() #sequential model: 순차적 (위->아래)

#input layer
model.add(Dense(3, input_dim=1)) #keras's dense 쌓기, input 차원은 1개  

#hidden layer
#hyper parameter tuning 가능(input, output layer는 불가)
model.add(Dense(30)) #dense 30개
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(40)) 
model.add(Dense(10))

#output layer
model.add(Dense(1)) # =input_dim


#3. 컴파일, 훈련 
# # mse  구하는 법 숙지
# 손실값을 mse로 잡겠다, 최적화를 adam으로 하겠다, 평가지표는 accuracy (영향은 x)
model.compile(loss='mse', optimizer='adam', metrics=['acc']) # hyper parameter tuning


model.fit(x, y, epochs=100, batch_size=1) # hyper parameter tuning

# 모델 훈련 (정제 데이터 머신에게), 100번, 1개씩 잘라서 작업 1 2 3 4 5 이렇게

#4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=1)


print("loss : ", loss)
print("acc : ", acc)
