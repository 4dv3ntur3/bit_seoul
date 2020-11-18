#2020-11-18 (8일차)
#load_breast_cancer -> DNN: load_model
#1) 모델&가중치(fit 이후 save한 모델 / ModelCheckPoint / 가중치)




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



############## 1. load_model #############################

#3. 컴파일, 훈련

from tensorflow.keras.models import load_model
model1 = load_model('./save/cancer_dnn_model_weigths.h5')

#4. 평가, 예측
result1 = model1.evaluate(x_test, y_test, batch_size=32)
print("====model & weights 같이 저장=========")
print("loss : ", result1[0])
print("accuracy : ", result1[1])


############## 2. load_model ModelCheckPoint #############

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


model3 = Sequential()
model3.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
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