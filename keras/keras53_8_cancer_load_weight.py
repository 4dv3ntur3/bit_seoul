#2020-11-18 (8일차)
#load_breast_cancer -> DNN + ModelCheckPoint
#유방암 데이터 -> 걸렸는지 / 안 걸렸는지 -> DNN
#Classes 2

#2진 분류

#sigmoid(relu, softmax 이전에 나왔던 것) - MinMaxScaler와 유사
# 함수값이 (0, 1)로 제한된다.
# 중간 값은 1/2이다.
# 매우 큰 값을 가지면 함수값은 거의 1이며, 매우 작은 값을 가지면 거의 0이다.

#단점
# 1) Gradient Vanishing 현상:
# 미분함수에 대해 x=0에서 최대값 1/4 을 가지고, input값이 일정이상 올라가면 미분값이 거의 0에 수렴하게된다. 
# 이는 |x|값이 커질 수록 Gradient Backpropagation시 미분값이 소실될 가능성이 크다.
# 2) 함수값 중심이 0이 아니다. : 함수값 중심이 0이 아니라 학습이 느려질 수 있다. 
# 만약 모든 x값들이 같은 부호(ex. for all x is positive) 라고 가정하고 아래의 파라미터 w에 대한 미분함수식을 살펴보자. 
# ∂L∂w=∂L∂a∂a∂w 그리고 ∂a∂w=x이기 때문에, ∂L∂w=∂L∂ax 이다. 위 식에서 모든 x가 양수라면 결국 ∂L∂w는 ∂L∂a 부호에 의해 결정된다.
# 따라서 한 노드에 대해 모든 파라미터w의 미분값은 모두 같은 부호를 같게된다. 
# 따라서 같은 방향으로 update되는데 이러한 과정은 학습을 zigzag 형태로 만들어 느리게 만드는 원인이 된다.
# exp 함수 사용시 비용이 크다.







import numpy as np
from sklearn.datasets import load_iris

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D, Dropout



#1. 데이터 
from sklearn.datasets import load_breast_cancer

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(x.shape) #(569, 30)
print(y.shape) #(569,)




#전처리
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7
)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)


x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# print(x_train.shape)


#========= 1. load_model (fit 이후 save 모델) ===================
#3. 컴파일, 훈련

from tensorflow.keras.models import load_model
model1 = load_model('./save/cancer_dnn_model_weigths.h5')

#4. 평가, 예측
result1 = model1.evaluate(x_test, y_test, batch_size=32)
print("====model & weights 같이 저장=========")
print("loss : ", result1[0])
print("accuracy : ", result1[1])


############## 2. load_model ModelCheckPoint #############

from tensorflow.keras.models import load_model
model2 = load_model('./model/cancer_dnn_44-0.0016.hdf5')

#4. 평가, 예측

result2 = model2.evaluate(x_test, y_test, batch_size=32)
print("=======checkpoint 저장=========")
print("loss : ", result2[0])
print("accuracy : ", result2[1])


################ 3. load_weights ##################


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout #LSTM도 layer


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, MaxPooling2D, Flatten #LSTM도 layer


model3 = Sequential()
model3.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],))) #flatten하면서 곱하고 dense에서 또 100 곱함 
                                        #Conv2d의 activatio n default='relu'
                                        #LSTM의 activation default='tanh'
model3.add(Dense(512, activation='relu'))
model3.add(Dropout(0.5))
model3.add(Dense(256, activation='relu'))
model3.add(Dropout(0.5))
model3.add(Dense(128, activation='relu'))
model3.add(Dense(300, activation='relu'))
model3.add(Dense(1024, activation='relu'))
model3.add(Dense(150, activation='relu'))
model3.add(Dense(70, activation='relu'))
model3.add(Dense(32, activation='relu'))
model3.add(Dense(1, activation='sigmoid')) #2진분류: sigmoid -> output: 0 or 1 이니까 1개임 


# model.summary()

# 3. 컴파일

model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model3.load_weights('./save/cancer_dnn_weights.h5')


#4. 평가, 예측
result3 = model3.evaluate(x_test, y_test, batch_size=32)
print("========weights 저장=========")
print("loss : ", result3[0])
print("accuracy : ", result3[1])



'''
====model & weights 같이 저장=========
loss :  0.013032753020524979
accuracy :  0.9941520690917969
6/6 [==============================] - 0s 1ms/step - loss: 0.0990 - accuracy: 0.9825
=======checkpoint 저장=========
loss :  0.09902040660381317
accuracy :  0.9824561476707458
6/6 [==============================] - 0s 1ms/step - loss: 7.3199e-08 - acc: 0.9942
========weights 저장=========
loss :  7.319869865796136e-08
accuracy :  0.9941520690917969

'''