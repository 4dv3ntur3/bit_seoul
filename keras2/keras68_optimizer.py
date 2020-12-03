#2020-12-02 (18일차)
#learning rate 
#gradient descent (경사하강법)

import numpy as np

#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense


#2. 모델 구성 
model = Sequential() 

model.add(Dense(30, input_dim=1)) 
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일, 훈련 
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad, RMSprop, SGD, Nadam


#lr = learning rate
optimizer = Adam(lr=0.001)  #loss :  0.00303988647647202     결과물:  [[10.95849]]

# optimizer = Adadelta(lr=0.001) #loss :  3.873276472091675       결과물:  [[7.5073943]]

# optimizer = Adamax(lr=0.001) #loss :  0.0021189148537814617   결과물:  [[10.9335985]]

# optimizer = Adagrad(lr=0.001) #loss :  0.0060004787519574165   결과물:  [[10.901627]

# optimizer = RMSprop(lr=0.001) #loss :  0.26574796438217163     결과물:  [[10.060978]]

#SGD : 통계적 경사하강법
# optimizer = SGD(lr=0.001) #loss :  0.00047661521239206195  결과물:  [[10.972857]]

# optimizer = Nadam(lr=0.001) #loss :  0.0013492617290467024   결과물:  [[10.953974]]





model.compile(loss='mse', optimizer=optimizer, metrics=['acc'])
model.fit(x, y, epochs=100) # hyper parameter tuning


#4. 평가, 예측
loss, mse = model.evaluate(x, y)


print()

y_pred = model.predict([11])
# 선형 회귀 모델에서는 accuracy를 사용할 수 없다
# 따라서 처음부터 accuracy라는 지표 자체가 틀렸다

print("loss : ", loss, "\t결과물: ", y_pred)



'''


'''
