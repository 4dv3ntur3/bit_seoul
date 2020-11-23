#2020-11-23 (11일차)
#Machine Learning: XOR + keras + hidden layer

from sklearn.svm import LinearSVC, SVC 
from sklearn.metrics import accuracy_score #분류모델. 회귀모델에서는 r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


#1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]

#2. 모델
# model = LinearSVC()
# model = SVC()


model = Sequential()
model.add(Dense(30, input_dim=2, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) #2진 분류면 one hot encoding 필요 없음. 3진 분류는 해야 됨
model.fit(x_data, y_data, batch_size=32, epochs=100)


#4. 평가, 예측

y_predict = model.predict(x_data) #y_data가 나오는지 보면 된다 


print(x_data, '의 예측 결과: ', y_predict)

acc_1 = model.evaluate(x_data, y_data)
print('model.evaluate: ', acc_1) #hidden layer 없으면 linearSVC랑 별차이 없음

# 사이킷런에서 제공하는 평가 지표니까 r2처럼 그대로 사용 가능 
# acc_2 = accuracy_score(y_data, y_predict)
# print("acc 2 : ", acc_2)



'''
hidden layer 추가 -> 1.0
두 번 세 번 연산해서 겨울 깼다 
1/1 [==============================] - 0s 1ms/step - loss: 0.0023 - acc: 1.0000
model.evaluate:  [0.002325830515474081, 1.0]
'''

